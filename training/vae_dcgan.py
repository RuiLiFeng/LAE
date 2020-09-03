# Copyright (c) 2019, NVIDIA Corporation. All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, visit
# https://nvlabs.github.io/stylegan2/license.html

"""Network architectures used in the StyleGAN2 paper."""

import numpy as np
import tensorflow as tf
import dnnlib
import dnnlib.tflib as tflib
from dnnlib.tflib.ops.upfirdn_2d import upsample_2d, downsample_2d, upsample_conv_2d, conv_downsample_2d
from dnnlib.tflib.ops.fused_bias_act import fused_bias_act

# NOTE: Do not import any application-specific modules here!
# Specify all network parameters as kwargs.

#----------------------------------------------------------------------------
# Get/create weight tensor for a convolution or fully-connected layer.

def get_weight(shape, gain=1, use_wscale=True, lrmul=1, weight_var='weight'):
    fan_in = np.prod(shape[:-1]) # [kernel, kernel, fmaps_in, fmaps_out] or [in, out]
    he_std = gain / np.sqrt(fan_in) # He init

    # Equalized learning rate and custom learning rate multiplier.
    if use_wscale:
        init_std = 1.0 / lrmul
        runtime_coef = he_std * lrmul
    else:
        init_std = he_std / lrmul
        runtime_coef = lrmul

    # Create variable.
    init = tf.initializers.random_normal(0, init_std)
    return tf.get_variable(weight_var, shape=shape, initializer=init) * runtime_coef

#----------------------------------------------------------------------------
# Fully-connected layer.

def dense_layer(x, fmaps, gain=1, use_wscale=True, lrmul=1, weight_var='weight'):
    if len(x.shape) > 2:
        x = tf.reshape(x, [-1, np.prod([d.value for d in x.shape[1:]])])
    w = get_weight([x.shape[1].value, fmaps], gain=gain, use_wscale=use_wscale, lrmul=lrmul, weight_var=weight_var)
    w = tf.cast(w, x.dtype)
    return tf.matmul(x, w)

#----------------------------------------------------------------------------
# Convolution layer with optional upsampling or downsampling.

def conv2d_layer(x, fmaps, kernel, up=False, down=False, resample_kernel=None, gain=1, use_wscale=True, lrmul=1, weight_var='weight'):
    assert not (up and down)
    assert kernel >= 1 and kernel % 2 == 1
    w = get_weight([kernel, kernel, x.shape[1].value, fmaps], gain=gain, use_wscale=use_wscale, lrmul=lrmul, weight_var=weight_var)
    if up:
        x = upsample_conv_2d(x, tf.cast(w, x.dtype), data_format='NCHW', k=resample_kernel)
    elif down:
        x = conv_downsample_2d(x, tf.cast(w, x.dtype), data_format='NCHW', k=resample_kernel)
    else:
        x = tf.nn.conv2d(x, tf.cast(w, x.dtype), data_format='NCHW', strides=[1,1,1,1], padding='SAME')
    return x

#----------------------------------------------------------------------------
# Apply bias and activation func.

def apply_bias_act(x, act='linear', alpha=None, gain=None, lrmul=1, bias_var='bias'):
    b = tf.get_variable(bias_var, shape=[x.shape[1]], initializer=tf.initializers.zeros()) * lrmul
    return fused_bias_act(x, b=tf.cast(b, x.dtype), act=act, alpha=alpha, gain=gain)

#----------------------------------------------------------------------------
# Naive upsampling (nearest neighbor) and downsampling (average pooling).

def naive_upsample_2d(x, factor=2):
    with tf.variable_scope('NaiveUpsample'):
        _N, C, H, W = x.shape.as_list()
        x = tf.reshape(x, [-1, C, H, 1, W, 1])
        x = tf.tile(x, [1, 1, 1, factor, 1, factor])
        return tf.reshape(x, [-1, C, H * factor, W * factor])

def naive_downsample_2d(x, factor=2):
    with tf.variable_scope('NaiveDownsample'):
        _N, C, H, W = x.shape.as_list()
        x = tf.reshape(x, [-1, C, H // factor, factor, W // factor, factor])
        return tf.reduce_mean(x, axis=[3,5])

#----------------------------------------------------------------------------
# Modulated convolution layer.

def modulated_conv2d_layer(x, y, fmaps, kernel, up=False, down=False, demodulate=True, resample_kernel=None, gain=1, use_wscale=True, lrmul=1, fused_modconv=True, weight_var='weight', mod_weight_var='mod_weight', mod_bias_var='mod_bias'):
    assert not (up and down)
    assert kernel >= 1 and kernel % 2 == 1

    # Get weight.
    w = get_weight([kernel, kernel, x.shape[1].value, fmaps], gain=gain, use_wscale=use_wscale, lrmul=lrmul, weight_var=weight_var)
    ww = w[np.newaxis] # [BkkIO] Introduce minibatch dimension.

    # Modulate.
    s = dense_layer(y, fmaps=x.shape[1].value, weight_var=mod_weight_var) # [BI] Transform incoming W to style.
    s = apply_bias_act(s, bias_var=mod_bias_var) + 1 # [BI] Add bias (initially 1).
    ww *= tf.cast(s[:, np.newaxis, np.newaxis, :, np.newaxis], w.dtype) # [BkkIO] Scale input feature maps.

    # Demodulate.
    if demodulate:
        d = tf.rsqrt(tf.reduce_sum(tf.square(ww), axis=[1,2,3]) + 1e-8) # [BO] Scaling factor.
        ww *= d[:, np.newaxis, np.newaxis, np.newaxis, :] # [BkkIO] Scale output feature maps.

    # Reshape/scale input.
    if fused_modconv:
        x = tf.reshape(x, [1, -1, x.shape[2], x.shape[3]]) # Fused => reshape minibatch to convolution groups.
        w = tf.reshape(tf.transpose(ww, [1, 2, 3, 0, 4]), [ww.shape[1], ww.shape[2], ww.shape[3], -1])
    else:
        x *= tf.cast(s[:, :, np.newaxis, np.newaxis], x.dtype) # [BIhw] Not fused => scale input activations.

    # Convolution with optional up/downsampling.
    if up:
        x = upsample_conv_2d(x, tf.cast(w, x.dtype), data_format='NCHW', k=resample_kernel)
    elif down:
        x = conv_downsample_2d(x, tf.cast(w, x.dtype), data_format='NCHW', k=resample_kernel)
    else:
        x = tf.nn.conv2d(x, tf.cast(w, x.dtype), data_format='NCHW', strides=[1,1,1,1], padding='SAME')

    # Reshape/scale output.
    if fused_modconv:
        x = tf.reshape(x, [-1, fmaps, x.shape[2], x.shape[3]]) # Fused => reshape convolution groups back to minibatch.
    elif demodulate:
        x *= tf.cast(d[:, :, np.newaxis, np.newaxis], x.dtype) # [BOhw] Not fused => scale output activations.
    return x

#----------------------------------------------------------------------------
# Minibatch standard deviation layer.

def minibatch_stddev_layer(x, group_size=4, num_new_features=1):
    group_size = tf.minimum(group_size, tf.shape(x)[0])     # Minibatch must be divisible by (or smaller than) group_size.
    s = x.shape                                             # [NCHW]  Input shape.
    y = tf.reshape(x, [group_size, -1, num_new_features, s[1]//num_new_features, s[2], s[3]])   # [GMncHW] Split minibatch into M groups of size G. Split channels into n channel groups c.
    y = tf.cast(y, tf.float32)                              # [GMncHW] Cast to FP32.
    y -= tf.reduce_mean(y, axis=0, keepdims=True)           # [GMncHW] Subtract mean over group.
    y = tf.reduce_mean(tf.square(y), axis=0)                # [MncHW]  Calc variance over group.
    y = tf.sqrt(y + 1e-8)                                   # [MncHW]  Calc stddev over group.
    y = tf.reduce_mean(y, axis=[2,3,4], keepdims=True)      # [Mn111]  Take average over fmaps and pixels.
    y = tf.reduce_mean(y, axis=[2])                         # [Mn11] Split channels into c channel groups
    y = tf.cast(y, x.dtype)                                 # [Mn11]  Cast back to original data type.
    y = tf.tile(y, [group_size, 1, s[2], s[3]])             # [NnHW]  Replicate over group and pixels.
    return tf.concat([x, y], axis=1)                        # [NCHW]  Append as new fmap.

#----------------------------------------------------------------------------
# Main generator network.
# Composed of two sub-networks (mapping and synthesis) that are defined below.
# Used in configs B-F (Table 1).

def Decoder_main(
    latents_in,                                         # First input: Latent vectors (Z) [minibatch, latent_size].
    labels_in,                                          # Second input: Conditioning labels [minibatch, label_size].
    is_training             = False,                    # Network is under training? Enables and disables specific features.
    return_dlatents         = False,                    # Return dlatents in addition to the images?
    is_template_graph       = False,                    # True = template graph constructed by the Network class, False = actual evaluation.
    components              = dnnlib.EasyDict(),        # Container for sub-networks. Retained between calls.
    mapping_func            = 'Decoder_mapping',              # Build func name for the mapping network.
    synthesis_func          = 'Decoder_synthesis',  # Build func name for the synthesis network.
    **kwargs
):
    # Setup components.
    if 'synthesis' not in components:
        components.synthesis = tflib.Network('G_synthesis', func_name=globals()[synthesis_func], **kwargs)
    if 'mapping' not in components:
        components.mapping = tflib.Network('G_mapping', func_name=globals()[mapping_func], **kwargs)

    # Evaluate mapping network.
    dlatents = components.mapping.get_output_for(latents_in, labels_in, is_training=is_training, **kwargs)
    dlatents = tf.cast(dlatents, tf.float32)

    images_out = components.synthesis.get_output_for(dlatents, is_training=is_training,
                                                     force_clean_graph=is_template_graph, **kwargs)

    # Return requested outputs.
    images_out = tf.identity(images_out, name='images_out')
    if return_dlatents:
        return images_out, dlatents
    return images_out


def Decoder_mapping(
        dlatents_in,
        labels_in,
        label_size           = 0,
        dlatent_size         = 512,
        dtype                = 'float32',
        act='lrelu',
        **kwargs
):
    dlatents_in.set_shape([None, dlatent_size])
    labels_in.set_shape([None, label_size])
    latents_in = tf.cast(dlatents_in, dtype)
    labels_in = tf.cast(labels_in, dtype)

    return tf.identity(dlatents_in, name='dlatents_in')


def Decoder_synthesis(
        dlatents_in,
        dlatent_size    = 512,
        num_channels    = 3,
        resolution      = 128,
        dtype           = 'float32',
        num_units       = 1024,
        resample_kernel = [1,3,3,1],
        is_training     = True,
        act             = 'lrelu',
        **kwargs
):
    # Primary inputs.
    dlatents_in.set_shape([None, dlatent_size])
    dlatents_in = tf.cast(dlatents_in, dtype)

    resolution_log2 = int(np.log2(resolution))
    assert resolution == 2 ** resolution_log2 and resolution >= 4
    num_layers = resolution_log2 - 2

    height = resolution // 2 ** (num_layers - 1)
    width = resolution // 2 ** (num_layers - 1)

    with tf.variable_scope('Dense'):
        z = linear(dlatents_in, num_units * height * width)
        z = tf.reshape(z, [-1, num_units, height, width])
        z = tf.nn.relu(z)

    for layer_id in range(num_layers - 1):
        with tf.variable_scope('conv%d' % layer_id):
            scale = 2 ** (layer_id + 1)
            _out_shape = [tf.shape(z)[0], num_units // scale, height * scale,
                          width * scale]
            z = deconv2d(z, _out_shape, stddev=0.0099999, conv_filters_dim=5)
            z = tf.layers.batch_normalization(z, training=is_training)
            z = tf.nn.relu(z)

    with tf.variable_scope('toRGB'):
        z = deconv2d(z, [tf.shape(z)[0], num_channels, resolution,
                         resolution], stddev=0.0099999, d_h=1, d_w=1, conv_filters_dim=5)
        images_out = tf.nn.tanh(z)

    return tf.identity(images_out, name='images_out')


def Encoder(
        images_in,
        labels_in,
        label_size       = 0,
        dlatent_size     = 512,
        num_channels     = 3,
        resolution       = 128,
        num_units        = 1024,
        dtype            = 'float32',
        is_training      = True,
        resample_kernel  = [1,3,3,1],
        act              = 'lrelu',
        **kwargs
):
    images_in.set_shape([None, num_channels, resolution, resolution])
    images_in = tf.cast(images_in, dtype)

    labels_in.set_shape([None, label_size])
    labels_in = tf.cast(labels_in, dtype)

    resolution_log2 = int(np.log2(resolution))
    assert resolution == 2 ** resolution_log2 and resolution >= 4
    num_layers = resolution_log2 - 2
    x = images_in

    for layer_id in range(num_layers):
        with tf.variable_scope('conv%d' % layer_id):
            scale = 2 ** (num_layers - layer_id - 1)
            x = conv2d(x, num_units // scale, k_w=5, k_h=5, d_h=2, d_w=2, stddev=0.0099999)
            x = tf.layers.batch_normalization(x, training=is_training)
            x = tf.nn.relu(x)

    x = tf.reshape(x, [-1, np.prod(x.shape[1:])])

    with tf.variable_scope('mu'):
        mu = linear(x, dlatent_size)
    with tf.variable_scope('log_sigma'):
        log_sigma = linear(x, dlatent_size)
    with tf.variable_scope('reparametric'):
        dlatents_out = reparametric(mu, log_sigma)

    return tf.identity(dlatents_out, name='dlatents_out'), \
           tf.identity(mu, name='mu'), tf.identity(log_sigma, name='log_sigma')


def reparametric(mu, log_sigma, distribution='normal', name=None):
    sigma = tf.exp(log_sigma * 0.5)
    if distribution == 'normal':
        epi = tf.random.normal(tf.shape(mu), dtype=mu.dtype)
    else:
        raise ValueError('Not supported distribution type %s !' % distribution)
    if name is not None:
        z = tf.add(tf.multiply(epi, sigma), mu, name=name)
    else:
        z = tf.multiply(epi, sigma) + mu
    return z


def deconv2d(input_, output_shape, d_h=2, d_w=2, stddev=0.02, scope=None, conv_filters_dim=None, padding='SAME'):
    """Transposed convolution (fractional stride convolution) layer.
    """

    shape = input_.get_shape().as_list()
    k_h = conv_filters_dim
    k_w = k_h

    assert len(shape) == 4, 'Conv2d_transpose works only with 4d tensors.'
    assert len(output_shape) == 4, 'outut_shape should be 4dimensional'

    with tf.variable_scope(scope or "deconv2d"):
        w = tf.get_variable(
            'filter', [k_h, k_w, output_shape[1], shape[1]],
            initializer=tf.random_normal_initializer(stddev=stddev))
        deconv = tf.nn.conv2d_transpose(
            input_, w, output_shape=output_shape,
            strides=[1, 1, d_h, d_w], padding=padding,  data_format="NCHW")
        deconv = apply_bias_act(deconv)
    return deconv


def conv2d(inputs, output_dim, k_h, k_w, d_h, d_w, stddev=0.02, name="conv2d",
           use_sn=False, use_bias=True):
    """Performs 2D convolution of the input."""
    with tf.variable_scope(name):
        w = tf.get_variable(
            "kernel", [k_h, k_w, inputs.shape[1].value, output_dim],
            initializer=weight_initializer(stddev=stddev))
        outputs = tf.nn.conv2d(inputs, w, strides=[1, 1, d_h, d_w], padding="SAME", data_format="NCHW")
        if use_bias:
            outputs = apply_bias_act(outputs)
    return outputs


def weight_initializer(initializer="orthogonal", stddev=0.02):
    """Returns the initializer for the given name.

    Args:
      initializer: Name of the initalizer. Use one in consts.INITIALIZERS.
      stddev: Standard deviation passed to initalizer.

    Returns:
      Initializer from `tf.initializers`.
    """
    if initializer == "normal":
        return tf.initializers.random_normal(stddev=stddev)
    if initializer == "truncated":
        return tf.initializers.truncated_normal(stddev=stddev)
    if initializer == "orthogonal":
        return tf.initializers.orthogonal()
    raise ValueError("Unknown weight initializer {}.".format(initializer))


def linear(inputs, output_size, scope=None, stddev=0.02, bias_start=0.0,
           use_sn=False, use_bias=True):
    """Linear layer without the non-linear activation applied."""
    shape = inputs.get_shape().as_list()
    with tf.variable_scope(scope or "linear"):
        kernel = tf.get_variable(
            "kernel",
            [shape[1], output_size],
            initializer=weight_initializer(stddev=stddev))
        outputs = tf.matmul(inputs, kernel)
        if use_bias:
            bias = tf.get_variable(
                "bias",
                [output_size],
                initializer=tf.constant_initializer(bias_start))
            outputs += bias
        return outputs
# Virtual Array Synthesis Architecture
#
# VASA
#
# Miro Ronac Giannone (mronacgiannone@smu.edu)
#-----------------------------------------------------------------------------------------------------------------#
# Import pacakages
import warnings, sys
sys.path.append('/Path/to/Cardinal')
import cardinal, cardinal_fk
#-----------------------------------------------------------------------------------------------------------------#
# Import packages as
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
#-----------------------------------------------------------------------------------------------------------------#
# Import functions from packages
from obspy import *
from obspy.core import *
from pathlib import Path
from obspy.geodetics import gps2dist_azimuth
#-----------------------------------------------------------------------------------------------------------------#
# ML Packages 
import tensorflow as tf
from keras.utils import *
from tensorflow.keras import *
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import Adam
#-----------------------------------------------------------------------------------------------------------------#
# Ignore non-critical warnings
warnings.filterwarnings("ignore")

'------------------------------------------------------------------------------------------------------------------------------------------------------------------------'
# Distance matrix construction
def load_site_file(site_path):
    """
    Load .site file with columns:
    station latitude longitude elevation

    Assumes elevation is in km
    """

    site_df = pd.read_csv(
        site_path,
        delim_whitespace=True,
        names=["station", "latitude", "longitude", "elevation_km"]
    )

    return site_df

def build_distance_matrix_from_site(site_path,
                                    station_order,
                                    use_elevation=False,
                                    normalize=True):
    """
    Returns:
        D: [S, S] float32 distance matrix.

    If normalize=True, distances are scaled to [0, 1] by max distance.
    """

    site_df = load_site_file(site_path)
    site_df = site_df.set_index("station").loc[station_order]

    n = len(station_order)
    D_km = np.zeros((n, n), dtype=np.float32)

    for i, sta_i in enumerate(station_order):
        lat_i = site_df.loc[sta_i, "latitude"]
        lon_i = site_df.loc[sta_i, "longitude"]
        elev_i_km = site_df.loc[sta_i, "elevation_km"]

        for j, sta_j in enumerate(station_order):
            lat_j = site_df.loc[sta_j, "latitude"]
            lon_j = site_df.loc[sta_j, "longitude"]
            elev_j_km = site_df.loc[sta_j, "elevation_km"]

            horizontal_m, _, _ = gps2dist_azimuth(lat_i, lon_i,
                                                  lat_j, lon_j)

            horizontal_km = horizontal_m / 1000.0 # convert to km

            if use_elevation:
                dz_km = elev_i_km - elev_j_km
                distance_km = np.sqrt(horizontal_km**2 + dz_km**2)
            else:
                distance_km = horizontal_km

            D_km[i, j] = distance_km

    if normalize:
        D = D_km / (D_km.max() + 1e-8)
    else:
        D = D_km

    return D.astype(np.float32), D_km.astype(np.float32), site_df.reset_index()

'------------------------------------------------------------------------------------------------------------------------------------------------------------------------'
# Scale using global asinh per component
def compute_global_component_mad_scale(X_train, n_events=3000, eps=1e-12, seed=123):
    """
    Robust per-component center/scale using median and MAD.
    X_train: [N, S, T, C]
    """

    rng = np.random.default_rng(seed)

    n = X_train.shape[0]
    idx = rng.choice(n, size=min(n_events, n), replace=False)

    x = X_train[idx].astype(np.float32)
    x_flat = x.reshape(-1, x.shape[-1])

    center = np.median(x_flat, axis=0)

    abs_dev = np.abs(x_flat - center[None, :])
    mad = np.median(abs_dev, axis=0)

    # 1.4826 makes MAD comparable to std for Gaussian data.
    scale = 1.4826 * mad
    scale = np.maximum(scale, eps)

    return center.astype(np.float32), scale.astype(np.float32)
    
# Scale using global asinh over all components
def compute_global_shared_mad_scale(X_train, n_events=3000, eps=1e-12, seed=123):
    """
    Robust shared center/scale using median and MAD.

    X_train: [N, S, T, C]

    Returns:
        center: scalar
        scale:  scalar

    This preserves relative amplitude differences between components better
    than per-component scaling.
    """

    rng = np.random.default_rng(seed)

    n = X_train.shape[0]
    idx = rng.choice(n, size=min(n_events, n), replace=False)

    x = X_train[idx].astype(np.float32)

    center = np.median(x)

    abs_dev = np.abs(x - center)
    mad = np.median(abs_dev)

    scale = 1.4826 * mad
    scale = max(scale, eps)

    return np.float32(center), np.float32(scale)

def global_asinh_scale(
    x_clean,
    global_center,
    global_scale,
    output_gain=1.0,
    final_clip=8.0
):
    """
    x_clean: [S, T, C]

    global_center/global_scale can be:
        scalar: shared across all components
        [C]:    per-component
    """

    global_center = tf.cast(global_center, tf.float32)
    global_scale = tf.cast(global_scale, tf.float32)

    z = (x_clean - global_center) / global_scale
    x_scaled = tf.asinh(z) * output_gain

    if final_clip is not None:
        x_scaled = tf.clip_by_value(x_scaled, -final_clip, final_clip)

    return x_scaled

def global_signed_log1p_scale(
    x_clean,
    global_center,
    global_scale,
    output_gain=1.0,
    final_clip=8.0
):
    global_center = tf.cast(global_center, tf.float32)
    global_scale = tf.cast(global_scale, tf.float32)

    z = (x_clean - global_center) / global_scale

    x_scaled = tf.sign(z) * tf.math.log1p(tf.abs(z))
    x_scaled = x_scaled * output_gain

    if final_clip is not None:
        x_scaled = tf.clip_by_value(x_scaled, -final_clip, final_clip)

    return x_scaled

'------------------------------------------------------------------------------------------------------------------------------------------------------------------------'
# Augmentations
# Noise
def add_observed_noise_percent(x_masked, observed, pct=1.0, eps=1e-12):
    """
    Add Gaussian noise to observed waveform samples only.

    x_masked: [S, T, C]
        Scaled waveform input with masked sensor already zeroed.

    observed: [S, T, C]
        1 where observed, 0 where masked.

    pct:
        Noise level as percent of per-event/per-component observed std.
        Example: pct=1.0 means noise std = 1% of observed waveform std.
    """

    observed_count = tf.reduce_sum(observed, axis=(0, 1), keepdims=True) + eps

    observed_mean = tf.reduce_sum(
        x_masked * observed,
        axis=(0, 1),
        keepdims=True
    ) / observed_count

    observed_var = tf.reduce_sum(
        tf.square(x_masked - observed_mean) * observed,
        axis=(0, 1),
        keepdims=True
    ) / observed_count

    observed_std = tf.sqrt(observed_var + eps)

    noise = tf.random.normal(
        shape=tf.shape(x_masked),
        mean=0.0,
        stddev=observed_std * (pct / 100.0),
        dtype=x_masked.dtype
    )

    # Add noise only to observed waveform samples, not masked target locations.
    return x_masked + noise * observed

# Adding random time shifts
def random_event_time_shift_zero_pad(x, max_shift_seconds=2.5, sampling_rate=40.0):
    max_shift_samples = int(round(max_shift_seconds * sampling_rate))

    if max_shift_samples <= 0:
        return x

    shift = tf.random.uniform(
        shape=[],
        minval=-max_shift_samples,
        maxval=max_shift_samples + 1,
        dtype=tf.int32
    )

    T = tf.shape(x)[1]

    def shift_right():
        pad = tf.zeros_like(x[:, :shift, :])
        return tf.concat([pad, x[:, :T - shift, :]], axis=1)

    def shift_left():
        shift_abs = -shift
        pad = tf.zeros_like(x[:, :shift_abs, :])
        return tf.concat([x[:, shift_abs:, :], pad], axis=1)

    x_shifted = tf.cond(
        shift > 0,
        shift_right,
        lambda: tf.cond(
            shift < 0,
            shift_left,
            lambda: x
        )
    )

    x_shifted.set_shape(x.shape)
    return x_shifted

# Stateless augmentations (same random pattern for each epoch)
def add_observed_noise_percent_stateless(
    x_masked,
    observed,
    index,
    seed,
    pct=1.0,
    eps=1e-12
):
    """
    Add Gaussian noise to observed waveform samples only.

    Fixed per event within an epoch if seed depends on epoch_seed and index.
    """
    x_masked = tf.cast(x_masked, tf.float32)
    observed = tf.cast(observed, tf.float32)
    index = tf.cast(index, tf.int32)
    seed = tf.cast(seed, tf.int32)

    observed_count = tf.reduce_sum(observed, axis=(0, 1), keepdims=True) + eps

    observed_mean = tf.reduce_sum(
        x_masked * observed,
        axis=(0, 1),
        keepdims=True
    ) / observed_count

    observed_var = tf.reduce_sum(
        tf.square(x_masked - observed_mean) * observed,
        axis=(0, 1),
        keepdims=True
    ) / observed_count

    observed_std = tf.sqrt(observed_var + eps)

    noise = tf.random.stateless_normal(
        shape=tf.shape(x_masked),
        seed=tf.stack([seed + 20000, index]),
        mean=0.0,
        stddev=1.0,
        dtype=x_masked.dtype
    )

    noise = noise * observed_std * (pct / 100.0)

    return x_masked + noise * observed

def random_signal_gain_augmentation_stateless(
    x_clean,
    index,
    seed,
    min_gain=1.0,
    max_gain=3.0,
    onset_start_s=10.0,
    onset_full_s=12.5,
    sampling_rate=40.0
):
    """
    Apply a smooth gain ramp so pre-event noise stays near original level
    and the later signal window is amplified.

    Gain is sampled log-uniformly, which is better for multiplicative
    amplitude variability like earthquakes.

    Fixed per event within an epoch if seed depends on epoch_seed and index.
    """
    x_clean = tf.cast(x_clean, tf.float32)
    index = tf.cast(index, tf.int32)
    seed = tf.cast(seed, tf.int32)

    min_gain = tf.cast(min_gain, tf.float32)
    max_gain = tf.cast(max_gain, tf.float32)

    T = tf.shape(x_clean)[1]
    t = tf.cast(tf.range(T), tf.float32) / sampling_rate

    log10_min = tf.math.log(min_gain) / tf.math.log(tf.constant(10.0, dtype=tf.float32))
    log10_max = tf.math.log(max_gain) / tf.math.log(tf.constant(10.0, dtype=tf.float32))

    log10_gain = tf.random.stateless_uniform(
        shape=[],
        seed=tf.stack([seed + 10000, index]),
        minval=log10_min,
        maxval=log10_max,
        dtype=tf.float32
    )

    gain = tf.pow(tf.constant(10.0, dtype=tf.float32), log10_gain)

    ramp = (t - onset_start_s) / (onset_full_s - onset_start_s)
    ramp = tf.clip_by_value(ramp, 0.0, 1.0)

    gain_curve = 1.0 + (gain - 1.0) * ramp
    gain_curve = gain_curve[None, :, None]

    return x_clean * gain_curve


def random_event_time_shift_zero_pad_stateless(
    x,
    index,
    seed,
    max_shift_seconds=2.5,
    sampling_rate=40.0
):
    """
    Apply one random zero-padded time shift to the whole event.

    Fixed per event within an epoch if seed depends on epoch_seed and index.
    """
    x = tf.cast(x, tf.float32)
    index = tf.cast(index, tf.int32)
    seed = tf.cast(seed, tf.int32)

    max_shift_samples = int(round(max_shift_seconds * sampling_rate))

    if max_shift_samples <= 0:
        return x

    shift = tf.random.stateless_uniform(
        shape=[],
        seed=tf.stack([seed, index]),
        minval=-max_shift_samples,
        maxval=max_shift_samples + 1,
        dtype=tf.int32
    )

    T = tf.shape(x)[1]

    def shift_right():
        pad = tf.zeros_like(x[:, :shift, :])
        return tf.concat([pad, x[:, :T - shift, :]], axis=1)

    def shift_left():
        shift_abs = -shift
        pad = tf.zeros_like(x[:, :shift_abs, :])
        return tf.concat([x[:, shift_abs:, :], pad], axis=1)

    x_shifted = tf.cond(
        shift > 0,
        shift_right,
        lambda: tf.cond(
            shift < 0,
            shift_left,
            lambda: x
        )
    )

    x_shifted.set_shape(x.shape)
    return x_shifted
    
'------------------------------------------------------------------------------------------------------------------------------------------------------------------------'
# Construct datasets with fixed random sensors per epoch
AUTOTUNE = tf.data.AUTOTUNE
def make_fixed_sensor_example_global_scaled(
    index,
    x_clean,
    distance_matrix,
    global_center,
    global_scale,
    seed=1234,
    noise_pct=0.0,
    output_gain=1.0,
    final_clip=8.0,
    scale_type="log1p",
    component_index=None,
    eps=1e-12
):
    x_clean = tf.cast(x_clean, tf.float32)
    distance_matrix = tf.cast(distance_matrix, tf.float32)

    sensor_dim = tf.shape(x_clean)[0]
    temporal_dim = tf.shape(x_clean)[1]
    channel_dim = tf.shape(x_clean)[2]

    index = tf.cast(index, tf.int32)

    sensor_id = tf.random.stateless_uniform(
        shape=[],
        seed=tf.stack([tf.cast(seed, tf.int32), index]),
        minval=0,
        maxval=sensor_dim,
        dtype=tf.int32
    )

    sensor_mask = tf.one_hot(sensor_id, sensor_dim, dtype=tf.float32)[:, None, None]
    mask = tf.tile(sensor_mask, [1, temporal_dim, channel_dim])

    observed = 1.0 - mask

    if scale_type == "asinh":
        x_scaled = global_asinh_scale(
            x_clean,
            global_center=global_center,
            global_scale=global_scale,
            output_gain=output_gain,
            final_clip=final_clip
        )
    elif scale_type == "log1p":
        x_scaled = global_signed_log1p_scale(
            x_clean,
            global_center=global_center,
            global_scale=global_scale,
            output_gain=output_gain,
            final_clip=final_clip
        )
    else:
        raise ValueError(f"Unknown scale_type: {scale_type}")

    if component_index is not None:
        x_scaled = x_scaled[..., component_index:component_index + 1]
        mask = mask[..., component_index:component_index + 1]
        observed = observed[..., component_index:component_index + 1]


    x_masked = x_scaled * observed

    if noise_pct > 0.0:
        x_masked = add_observed_noise_percent(
            x_masked,
            observed,
            pct=noise_pct,
            eps=eps
        )

    x_model = tf.concat([x_masked, observed], axis=-1)
    y_with_mask = tf.concat([x_scaled, mask], axis=-1)

    inputs = {
        "seismic_input": x_model,
        "distance_input": distance_matrix
    }

    return inputs, y_with_mask
    
def make_fixed_sensor_example_global_scaled_3branch(
    index,
    x_clean,
    distance_matrix,
    global_center,
    global_scale,
    seed=1234,
    noise_pct=0.0,
    max_shift_seconds=0.0,
    sampling_rate=40.0,
    output_gain=1.0,
    final_clip=8.0,
    scale_type="asinh",
    eps=1e-12
):
    x_clean = tf.cast(x_clean, tf.float32)
    distance_matrix = tf.cast(distance_matrix, tf.float32)

    if max_shift_seconds > 0.0:
        x_clean = random_event_time_shift_zero_pad(
            x_clean,
            max_shift_seconds=max_shift_seconds,
            sampling_rate=sampling_rate
        )

    sensor_dim = tf.shape(x_clean)[0]
    temporal_dim = tf.shape(x_clean)[1]
    channel_dim = tf.shape(x_clean)[2]

    index = tf.cast(index, tf.int32)

    sensor_id = tf.random.stateless_uniform(
        shape=[],
        seed=tf.stack([tf.cast(seed, tf.int32), index]),
        minval=0,
        maxval=sensor_dim,
        dtype=tf.int32
    )

    sensor_mask = tf.one_hot(sensor_id, sensor_dim, dtype=tf.float32)[:, None, None]
    mask = tf.tile(sensor_mask, [1, temporal_dim, channel_dim])

    observed = 1.0 - mask

    if scale_type == "asinh":
        x_scaled = global_asinh_scale(
            x_clean,
            global_center=global_center,
            global_scale=global_scale,
            output_gain=output_gain,
            final_clip=final_clip
        )
    elif scale_type == "log1p":
        x_scaled = global_signed_log1p_scale(
            x_clean,
            global_center=global_center,
            global_scale=global_scale,
            output_gain=output_gain,
            final_clip=final_clip
        )
    else:
        raise ValueError(f"Unknown scale_type: {scale_type}")

    x_masked = x_scaled * observed

    if noise_pct > 0.0:
        x_masked = add_observed_noise_percent(
            x_masked,
            observed,
            pct=noise_pct,
            eps=eps
        )

    x_model = tf.concat([x_masked, observed], axis=-1)

    targets = {
        "Z_Output": tf.concat([x_scaled[..., 0:1], mask[..., 0:1]], axis=-1),
        "N_Output": tf.concat([x_scaled[..., 1:2], mask[..., 1:2]], axis=-1),
        "E_Output": tf.concat([x_scaled[..., 2:3], mask[..., 2:3]], axis=-1),
    }

    inputs = {
        "seismic_input": x_model,
        "distance_input": distance_matrix
    }

    return inputs, targets

def make_fixed_sensor_example_global_scaled_3branch_stateless(
    index,
    x_clean,
    distance_matrix,
    global_center,
    global_scale,
    seed=1234,
    noise_pct=0.0,
    max_shift_seconds=0.0,
    use_signal_gain=False,
    min_gain=1.0,
    max_gain=3.0,
    onset_start_s=10.0,
    onset_full_s=12.5,
    sampling_rate=40.0,
    output_gain=1.0,
    final_clip=8.0,
    scale_type="asinh",
    eps=1e-12
):
    x_clean = tf.cast(x_clean, tf.float32)
    distance_matrix = tf.cast(distance_matrix, tf.float32)
    index = tf.cast(index, tf.int32)
    seed = tf.cast(seed, tf.int32)

    # 1. Signal gain augmentation in raw waveform space
    if use_signal_gain:
        x_clean = random_signal_gain_augmentation_stateless(
            x_clean,
            index=index,
            seed=seed,
            min_gain=min_gain,
            max_gain=max_gain,
            onset_start_s=onset_start_s,
            onset_full_s=onset_full_s,
            sampling_rate=sampling_rate
        )

    # 2. Time shift augmentation in raw waveform space
    if max_shift_seconds > 0.0:
        x_clean = random_event_time_shift_zero_pad_stateless(
            x_clean,
            index=index,
            seed=seed,
            max_shift_seconds=max_shift_seconds,
            sampling_rate=sampling_rate
        )

    sensor_dim = tf.shape(x_clean)[0]
    temporal_dim = tf.shape(x_clean)[1]
    channel_dim = tf.shape(x_clean)[2]

    # 3. Fixed masked sensor for this event within this epoch
    sensor_id = tf.random.stateless_uniform(
        shape=[],
        seed=tf.stack([seed, index]),
        minval=0,
        maxval=sensor_dim,
        dtype=tf.int32
    )

    sensor_mask = tf.one_hot(sensor_id, sensor_dim, dtype=tf.float32)[:, None, None]
    mask = tf.tile(sensor_mask, [1, temporal_dim, channel_dim])
    observed = 1.0 - mask

    # 4. Global scaling
    if scale_type == "asinh":
        x_scaled = global_asinh_scale(
            x_clean,
            global_center=global_center,
            global_scale=global_scale,
            output_gain=output_gain,
            final_clip=final_clip
        )
    elif scale_type == "log1p":
        x_scaled = global_signed_log1p_scale(
            x_clean,
            global_center=global_center,
            global_scale=global_scale,
            output_gain=output_gain,
            final_clip=final_clip
        )
    else:
        raise ValueError(f"Unknown scale_type: {scale_type}")

    x_masked = x_scaled * observed

    # 5. Noise augmentation on observed scaled waveform only
    if noise_pct > 0.0:
        x_masked = add_observed_noise_percent_stateless(
            x_masked,
            observed,
            index=index,
            seed=seed,
            pct=noise_pct,
            eps=eps
        )

    x_model = tf.concat([x_masked, observed], axis=-1)

    targets = {
        "Z_Output": tf.concat([x_scaled[..., 0:1], mask[..., 0:1]], axis=-1),
        "N_Output": tf.concat([x_scaled[..., 1:2], mask[..., 1:2]], axis=-1),
        "E_Output": tf.concat([x_scaled[..., 2:3], mask[..., 2:3]], axis=-1),
    }

    inputs = {
        "seismic_input": x_model,
        "distance_input": distance_matrix
    }

    return inputs, targets

def make_fixed_sensor_example_global_scaled_3branch_stateless_mag(
    index,
    x_clean,
    mag_value,
    distance_matrix,
    global_center,
    global_scale,
    seed=1234,
    noise_pct=0.0,
    max_shift_seconds=0.0,
    use_signal_gain=False,
    min_gain=1.0,
    max_gain=3.0,
    onset_start_s=10.0,
    onset_full_s=12.5,
    sampling_rate=40.0,
    output_gain=1.0,
    final_clip=8.0,
    scale_type="asinh",
    eps=1e-12
):
    x_clean = tf.cast(x_clean, tf.float32)
    mag_value = tf.cast(mag_value, tf.float32)
    distance_matrix = tf.cast(distance_matrix, tf.float32)
    index = tf.cast(index, tf.int32)
    seed = tf.cast(seed, tf.int32)

    # 1. Gain first
    if use_signal_gain:
        x_clean = random_signal_gain_augmentation_stateless(
            x_clean,
            index=index,
            seed=seed,
            min_gain=min_gain,
            max_gain=max_gain,
            onset_start_s=onset_start_s,
            onset_full_s=onset_full_s,
            sampling_rate=sampling_rate
        )

    # 2. Then time shift
    if max_shift_seconds > 0.0:
        x_clean = random_event_time_shift_zero_pad_stateless(
            x_clean,
            index=index,
            seed=seed,
            max_shift_seconds=max_shift_seconds,
            sampling_rate=sampling_rate
        )

    sensor_dim = tf.shape(x_clean)[0]
    temporal_dim = tf.shape(x_clean)[1]
    channel_dim = tf.shape(x_clean)[2]

    sensor_id = tf.random.stateless_uniform(
        shape=[],
        seed=tf.stack([seed + 30000, index]),
        minval=0,
        maxval=sensor_dim,
        dtype=tf.int32
    )

    sensor_mask = tf.one_hot(sensor_id, sensor_dim, dtype=tf.float32)[:, None, None]
    mask = tf.tile(sensor_mask, [1, temporal_dim, channel_dim])
    observed = 1.0 - mask

    if scale_type == "asinh":
        x_scaled = global_asinh_scale(
            x_clean,
            global_center=global_center,
            global_scale=global_scale,
            output_gain=output_gain,
            final_clip=final_clip
        )
    elif scale_type == "log1p":
        x_scaled = global_signed_log1p_scale(
            x_clean,
            global_center=global_center,
            global_scale=global_scale,
            output_gain=output_gain,
            final_clip=final_clip
        )
    else:
        raise ValueError(f"Unknown scale_type: {scale_type}")

    x_masked = x_scaled * observed

    if noise_pct > 0.0:
        x_masked = add_observed_noise_percent_stateless(
            x_masked,
            observed,
            index=index,
            seed=seed,
            pct=noise_pct,
            eps=eps
        )

    x_model = tf.concat([x_masked, observed], axis=-1)

    targets = {
        "Z_Output": tf.concat([x_scaled[..., 0:1], mask[..., 0:1]], axis=-1),
        "N_Output": tf.concat([x_scaled[..., 1:2], mask[..., 1:2]], axis=-1),
        "E_Output": tf.concat([x_scaled[..., 2:3], mask[..., 2:3]], axis=-1),
        "MAG_Output": tf.reshape(mag_value, (1,)),
    }

    inputs = {
        "seismic_input": x_model,
        "distance_input": distance_matrix,
    }

    return inputs, targets

'------------------------------------------------------------------------------------------------------------------------------------------------------------------------'
# Build training dataset
def make_vasa_epoch_train_dataset_global_scaled(
    X,
    D_norm,
    batch_size,
    global_center,
    global_scale,
    epoch_seed,
    shuffle_buffer=2048,
    noise_pct=0.0,
    output_gain=1.0,
    final_clip=8.0,
    scale_type="log1p",
    component_index=None
):
    D_norm = tf.constant(D_norm, dtype=tf.float32)
    global_center = tf.constant(global_center, dtype=tf.float32)
    global_scale = tf.constant(global_scale, dtype=tf.float32)

    ds = tf.data.Dataset.from_tensor_slices(X.astype(np.float32))
    ds = ds.enumerate()

    ds = ds.map(
        lambda i, x: make_fixed_sensor_example_global_scaled(
            i,
            x,
            D_norm,
            global_center,
            global_scale,
            seed=epoch_seed,
            noise_pct=noise_pct,
            output_gain=output_gain,
            final_clip=final_clip,
            scale_type=scale_type,
            component_index=component_index
        ),
        num_parallel_calls=tf.data.AUTOTUNE
    )

    ds = ds.shuffle(shuffle_buffer, reshuffle_each_iteration=True)
    ds = ds.batch(batch_size, drop_remainder=False)
    ds = ds.prefetch(tf.data.AUTOTUNE)

    return ds

# 3 separate branches
def make_vasa_epoch_train_dataset_global_scaled_3branch(
    X,
    D_norm,
    batch_size,
    global_center,
    global_scale,
    epoch_seed,
    shuffle_buffer=2048,
    noise_pct=0.0,
    max_shift_seconds=0.0,
    sampling_rate=40.0,
    output_gain=1.0,
    final_clip=8.0,
    scale_type="asinh"
):
    D_norm = tf.constant(D_norm, dtype=tf.float32)
    global_center = tf.constant(global_center, dtype=tf.float32)
    global_scale = tf.constant(global_scale, dtype=tf.float32)

    ds = tf.data.Dataset.from_tensor_slices(X.astype(np.float32))
    ds = ds.enumerate()

    ds = ds.map(
        lambda i, x: make_fixed_sensor_example_global_scaled_3branch(
            i,
            x,
            D_norm,
            global_center,
            global_scale,
            seed=epoch_seed,
            noise_pct=noise_pct,
            max_shift_seconds=max_shift_seconds,
            sampling_rate=sampling_rate,
            output_gain=output_gain,
            final_clip=final_clip,
            scale_type=scale_type
        ),
        num_parallel_calls=tf.data.AUTOTUNE
    )

    ds = ds.shuffle(shuffle_buffer, reshuffle_each_iteration=True)
    ds = ds.batch(batch_size, drop_remainder=False)
    ds = ds.prefetch(tf.data.AUTOTUNE)

    return ds

# Stateless augmentations (same random pattern per epoch)
def make_vasa_epoch_train_dataset_global_scaled_3branch_stateless(
    X,
    D_norm,
    batch_size,
    global_center,
    global_scale,
    epoch_seed,
    shuffle_buffer=2048,
    noise_pct=0.0,
    max_shift_seconds=0.0,
    use_signal_gain=False,
    min_gain=1.0,
    max_gain=3.0,
    onset_start_s=10.0,
    onset_full_s=12.5,
    sampling_rate=40.0,
    output_gain=1.0,
    final_clip=8.0,
    scale_type="asinh"
):
    D_norm = tf.constant(D_norm, dtype=tf.float32)
    global_center = tf.constant(global_center, dtype=tf.float32)
    global_scale = tf.constant(global_scale, dtype=tf.float32)

    ds = tf.data.Dataset.from_tensor_slices(X.astype(np.float32))
    ds = ds.enumerate()

    ds = ds.map(
        lambda i, x: make_fixed_sensor_example_global_scaled_3branch_stateless(
            i,
            x,
            D_norm,
            global_center,
            global_scale,
            seed=epoch_seed,
            noise_pct=noise_pct,
            max_shift_seconds=max_shift_seconds,
            use_signal_gain=use_signal_gain,
            min_gain=min_gain,
            max_gain=max_gain,
            onset_start_s=onset_start_s,
            onset_full_s=onset_full_s,
            sampling_rate=sampling_rate,
            output_gain=output_gain,
            final_clip=final_clip,
            scale_type=scale_type
        ),
        num_parallel_calls=tf.data.AUTOTUNE
    )

    ds = ds.shuffle(shuffle_buffer, reshuffle_each_iteration=True)
    ds = ds.batch(batch_size, drop_remainder=False)
    ds = ds.prefetch(tf.data.AUTOTUNE)

    return ds
    
def make_vasa_epoch_train_dataset_global_scaled_3branch_stateless_oversampled(
    X,
    meta_df,
    D_norm,
    batch_size,
    global_center,
    global_scale,
    epoch_seed,
    shuffle_buffer=2048,
    noise_pct=0.0,
    max_shift_seconds=0.0,
    use_signal_gain=False,
    min_gain=1.0,
    max_gain=3.0,
    onset_start_s=10.0,
    onset_full_s=12.5,
    sampling_rate=40.0,
    output_gain=1.0,
    final_clip=8.0,
    scale_type="asinh",
    mag_threshold=2.0,
    target_large_frac=0.25,
):
    X = np.asarray(X, dtype=np.float32)
    meta_df = meta_df.reset_index(drop=True)

    D_norm = tf.constant(D_norm, dtype=tf.float32)
    global_center = tf.constant(global_center, dtype=tf.float32)
    global_scale = tf.constant(global_scale, dtype=tf.float32)

    mags = meta_df["MAG"].to_numpy(dtype=np.float32)
    n_events = len(X)

    large_idx = np.where(mags >= mag_threshold)[0]
    other_idx = np.where(mags < mag_threshold)[0]

    if len(large_idx) == 0:
        raise ValueError(f"No events found with MAG >= {mag_threshold}")
    if len(other_idx) == 0:
        raise ValueError(f"No events found with MAG < {mag_threshold}")

    rng = np.random.default_rng(epoch_seed)

    n_large_target = int(round(target_large_frac * n_events))
    n_large_target = min(max(n_large_target, 1), n_events - 1)
    n_other_target = n_events - n_large_target

    sampled_large = rng.choice(large_idx, size=n_large_target, replace=True)
    replace_other = n_other_target > len(other_idx)
    sampled_other = rng.choice(other_idx, size=n_other_target, replace=replace_other)

    epoch_indices = np.concatenate([sampled_large, sampled_other])
    rng.shuffle(epoch_indices)

    X_epoch = X[epoch_indices]

    ds = tf.data.Dataset.from_tensor_slices(X_epoch)
    ds = ds.enumerate()

    ds = ds.map(
        lambda i, x: make_fixed_sensor_example_global_scaled_3branch_stateless(
            i,
            x,
            D_norm,
            global_center,
            global_scale,
            seed=epoch_seed,
            noise_pct=noise_pct,
            max_shift_seconds=max_shift_seconds,
            use_signal_gain=use_signal_gain,
            min_gain=min_gain,
            max_gain=max_gain,
            onset_start_s=onset_start_s,
            onset_full_s=onset_full_s,
            sampling_rate=sampling_rate,
            output_gain=output_gain,
            final_clip=final_clip,
            scale_type=scale_type
        ),
        num_parallel_calls=tf.data.AUTOTUNE
    )

    ds = ds.shuffle(shuffle_buffer, reshuffle_each_iteration=True)
    ds = ds.batch(batch_size, drop_remainder=False)
    ds = ds.prefetch(tf.data.AUTOTUNE)

    return ds, epoch_indices


def make_vasa_epoch_train_dataset_global_scaled_3branch_stateless_oversampled_mag(
    X,
    meta_df,
    D_norm,
    batch_size,
    global_center,
    global_scale,
    epoch_seed,
    shuffle_buffer=2048,
    noise_pct=0.0,
    max_shift_seconds=0.0,
    use_signal_gain=False,
    min_gain=1.0,
    max_gain=3.0,
    onset_start_s=10.0,
    onset_full_s=12.5,
    sampling_rate=40.0,
    output_gain=1.0,
    final_clip=8.0,
    scale_type="asinh",
    mag_threshold=2.0,
    target_large_frac=0.25,
):
    X = np.asarray(X, dtype=np.float32)
    meta_df = meta_df.reset_index(drop=True)

    D_norm = tf.constant(D_norm, dtype=tf.float32)
    global_center = tf.constant(global_center, dtype=tf.float32)
    global_scale = tf.constant(global_scale, dtype=tf.float32)

    mags = meta_df["MAG"].to_numpy(dtype=np.float32)
    n_events = len(X)

    large_idx = np.where(mags >= mag_threshold)[0]
    other_idx = np.where(mags < mag_threshold)[0]

    if len(large_idx) == 0:
        raise ValueError(f"No events found with MAG >= {mag_threshold}")
    if len(other_idx) == 0:
        raise ValueError(f"No events found with MAG < {mag_threshold}")

    rng = np.random.default_rng(epoch_seed)

    n_large_target = int(round(target_large_frac * n_events))
    n_large_target = min(max(n_large_target, 1), n_events - 1)
    n_other_target = n_events - n_large_target

    sampled_large = rng.choice(large_idx, size=n_large_target, replace=True)
    replace_other = n_other_target > len(other_idx)
    sampled_other = rng.choice(other_idx, size=n_other_target, replace=replace_other)

    epoch_indices = np.concatenate([sampled_large, sampled_other])
    rng.shuffle(epoch_indices)

    X_epoch = X[epoch_indices]
    mag_epoch = mags[epoch_indices]

    ds = tf.data.Dataset.from_tensor_slices((X_epoch, mag_epoch))
    ds = ds.enumerate()

    ds = ds.map(
        lambda i, data: make_fixed_sensor_example_global_scaled_3branch_stateless_mag(
            i,
            data[0],
            data[1],
            D_norm,
            global_center,
            global_scale,
            seed=epoch_seed,
            noise_pct=noise_pct,
            max_shift_seconds=max_shift_seconds,
            use_signal_gain=use_signal_gain,
            min_gain=min_gain,
            max_gain=max_gain,
            onset_start_s=onset_start_s,
            onset_full_s=onset_full_s,
            sampling_rate=sampling_rate,
            output_gain=output_gain,
            final_clip=final_clip,
            scale_type=scale_type
        ),
        num_parallel_calls=tf.data.AUTOTUNE
    )

    ds = ds.shuffle(shuffle_buffer, reshuffle_each_iteration=True)
    ds = ds.batch(batch_size, drop_remainder=False)
    ds = ds.prefetch(tf.data.AUTOTUNE)

    return ds, epoch_indices

'------------------------------------------------------------------------------------------------------------------------------------------------------------------------'
# Make test set
def make_vasa_val_dataset_global_scaled(
    X,
    D_norm,
    batch_size,
    global_center,
    global_scale,
    seed=2024,
    output_gain=1.0,
    final_clip=8.0,
    scale_type="log1p",
    component_index=None
):
    D_norm = tf.constant(D_norm, dtype=tf.float32)
    global_center = tf.constant(global_center, dtype=tf.float32)
    global_scale = tf.constant(global_scale, dtype=tf.float32)

    ds = tf.data.Dataset.from_tensor_slices(X.astype(np.float32))
    ds = ds.enumerate()

    ds = ds.map(
        lambda i, x: make_fixed_sensor_example_global_scaled(
            i,
            x,
            D_norm,
            global_center,
            global_scale,
            seed=seed,
            noise_pct=0.0,
            output_gain=output_gain,
            final_clip=final_clip,
            scale_type=scale_type,
            component_index=component_index
        ),
        num_parallel_calls=tf.data.AUTOTUNE
    )

    ds = ds.batch(batch_size, drop_remainder=False)
    ds = ds.prefetch(tf.data.AUTOTUNE)

    return ds

def make_vasa_val_dataset_global_scaled_3branch(
    X,
    meta_df,
    D_norm,
    batch_size,
    global_center,
    global_scale,
    seed=2024,
    output_gain=1.0,
    final_clip=8.0,
    scale_type="asinh",
):
    X = np.asarray(X, dtype=np.float32)

    D_norm = tf.constant(D_norm, dtype=tf.float32)
    global_center = tf.constant(global_center, dtype=tf.float32)
    global_scale = tf.constant(global_scale, dtype=tf.float32)

    ds = tf.data.Dataset.from_tensor_slices(X)
    ds = ds.enumerate()

    ds = ds.map(
        lambda i, x: make_fixed_sensor_example_global_scaled_3branch_stateless(
            i,
            x,
            D_norm,
            global_center,
            global_scale,
            seed=seed,
            noise_pct=0.0,
            max_shift_seconds=0.0,
            use_signal_gain=False,
            min_gain=1.0,
            max_gain=1.0,
            onset_start_s=10.0,
            onset_full_s=12.5,
            sampling_rate=40.0,
            output_gain=output_gain,
            final_clip=final_clip,
            scale_type=scale_type
        ),
        num_parallel_calls=tf.data.AUTOTUNE
    )

    ds = ds.batch(batch_size, drop_remainder=False)
    ds = ds.prefetch(tf.data.AUTOTUNE)

    return ds


def make_vasa_val_dataset_global_scaled_3branch_mag(
    X,
    meta_df,
    D_norm,
    batch_size,
    global_center,
    global_scale,
    seed=2024,
    output_gain=1.0,
    final_clip=8.0,
    scale_type="asinh",
):
    X = np.asarray(X, dtype=np.float32)
    meta_df = meta_df.reset_index(drop=True)
    mags = meta_df["MAG"].to_numpy(dtype=np.float32)

    D_norm = tf.constant(D_norm, dtype=tf.float32)
    global_center = tf.constant(global_center, dtype=tf.float32)
    global_scale = tf.constant(global_scale, dtype=tf.float32)

    ds = tf.data.Dataset.from_tensor_slices((X, mags))
    ds = ds.enumerate()

    ds = ds.map(
        lambda i, data: make_fixed_sensor_example_global_scaled_3branch_stateless_mag(
            i,
            data[0],
            data[1],
            D_norm,
            global_center,
            global_scale,
            seed=seed,
            noise_pct=0.0,
            max_shift_seconds=0.0,
            use_signal_gain=False,
            min_gain=1.0,
            max_gain=1.0,
            onset_start_s=10.0,
            onset_full_s=12.5,
            sampling_rate=40.0,
            output_gain=output_gain,
            final_clip=final_clip,
            scale_type=scale_type
        ),
        num_parallel_calls=tf.data.AUTOTUNE
    )

    ds = ds.batch(batch_size, drop_remainder=False)
    ds = ds.prefetch(tf.data.AUTOTUNE)

    return ds

'------------------------------------------------------------------------------------------------------------------------------------------------------------------------'
# Train dataset statistics
def sanity_check_batch(batch, mode="3branch"):
    inputs, y = batch
    x_model = inputs["seismic_input"]

    print("x_model:", x_model.shape)

    if mode == "3branch":
        x_masked = x_model[..., :3]
        observed_input = x_model[..., 3:]

        print("x_masked:", x_masked.shape)
        print("observed_input:", observed_input.shape)

        for output_name, comp_idx in [
            ("Z_Output", 0),
            ("N_Output", 1),
            ("E_Output", 2),
        ]:
            y_comp = y[output_name]

            y_true = y_comp[..., 0:1]
            mask = y_comp[..., 1:2]
            observed_comp = observed_input[..., comp_idx:comp_idx + 1]
            x_masked_comp = x_masked[..., comp_idx:comp_idx + 1]

            zero_pred = tf.zeros_like(y_true)
            zero_rmse = masked_1c_rmse_metric_2ch(y_comp, zero_pred)

            print(f"\n{output_name}")
            print("y_true:", y_true.shape)
            print("mask:", mask.shape)
            print("masked fraction:", tf.reduce_mean(mask).numpy())
            print("observed unique:", tf.unique(tf.reshape(observed_comp, [-1])).y.numpy())
            print("mask unique:", tf.unique(tf.reshape(mask, [-1])).y.numpy())
            print("max abs input where mask=1:", tf.reduce_max(tf.abs(x_masked_comp * mask)).numpy())
            print("max abs target where mask=1:", tf.reduce_max(tf.abs(y_true * mask)).numpy())
            print(
                "mean abs target where mask=1:",
                (tf.reduce_sum(tf.abs(y_true) * mask) / (tf.reduce_sum(mask) + 1e-8)).numpy()
            )
            print("zero baseline RMSE:", float(zero_rmse))
            print(
                "max abs observed + mask - 1:",
                tf.reduce_max(tf.abs(observed_comp - (1.0 - mask))).numpy()
            )

    elif mode == "single_component":
        x_masked = x_model[..., 0:1]
        observed_input = x_model[..., 1:2]

        y_true = y[..., 0:1]
        mask = y[..., 1:2]

        zero_pred = tf.zeros_like(y_true)
        zero_rmse = masked_1c_rmse_metric_2ch(y, zero_pred)

        print("x_masked:", x_masked.shape)
        print("observed_input:", observed_input.shape)
        print("y_true:", y_true.shape)
        print("mask:", mask.shape)
        print("masked fraction:", tf.reduce_mean(mask).numpy())
        print("observed unique:", tf.unique(tf.reshape(observed_input, [-1])).y.numpy())
        print("mask unique:", tf.unique(tf.reshape(mask, [-1])).y.numpy())
        print("max abs input where mask=1:", tf.reduce_max(tf.abs(x_masked * mask)).numpy())
        print("max abs target where mask=1:", tf.reduce_max(tf.abs(y_true * mask)).numpy())
        print(
            "mean abs target where mask=1:",
            (tf.reduce_sum(tf.abs(y_true) * mask) / (tf.reduce_sum(mask) + 1e-8)).numpy()
        )
        print("zero baseline RMSE:", float(zero_rmse))
        print(
            "max abs observed + mask - 1:",
            tf.reduce_max(tf.abs(observed_input - (1.0 - mask))).numpy()
        )

    elif mode == "full":
        x_masked = x_model[..., :3]
        observed_input = x_model[..., 3:]

        y_true = y[..., :3]
        mask = y[..., 3:]

        zero_pred = tf.zeros_like(y_true)
        zero_rmse = masked_rmse_metric(
            tf.concat([y_true, mask], axis=-1),
            zero_pred
        )

        print("x_masked:", x_masked.shape)
        print("observed_input:", observed_input.shape)
        print("y_true:", y_true.shape)
        print("mask:", mask.shape)
        print("masked fraction:", tf.reduce_mean(mask).numpy())
        print("observed unique:", tf.unique(tf.reshape(observed_input, [-1])).y.numpy())
        print("mask unique:", tf.unique(tf.reshape(mask, [-1])).y.numpy())
        print("max abs input where mask=1:", tf.reduce_max(tf.abs(x_masked * mask)).numpy())
        print("max abs target where mask=1:", tf.reduce_max(tf.abs(y_true * mask)).numpy())
        print(
            "mean abs target where mask=1:",
            (tf.reduce_sum(tf.abs(y_true) * mask) / (tf.reduce_sum(mask) + 1e-8)).numpy()
        )
        print("zero baseline RMSE:", float(zero_rmse))
        print(
            "max abs observed + mask - 1:",
            tf.reduce_max(tf.abs(observed_input - (1.0 - mask))).numpy()
        )

    else:
        raise ValueError("mode must be '3branch', 'single_component', or 'full'")

def print_clip_fraction_from_batch(batch, mode="3branch", clip_value=8.0):
    inputs, y = batch

    def compute_clip_stats(y_true, mask, label):
        clip_hits = tf.cast(
            tf.abs(y_true) >= clip_value - 1e-6,
            tf.float32
        )

        overall_clip_fraction = tf.reduce_mean(clip_hits)

        masked_clip_fraction = (
            tf.reduce_sum(clip_hits * mask) /
            (tf.reduce_sum(mask) + 1e-8)
        )

        overall_clip_fraction = float(overall_clip_fraction.numpy())
        masked_clip_fraction = float(masked_clip_fraction.numpy())

        print(f"\n{label}")
        print("y_true:", y_true.shape)
        print("mask:", mask.shape)
        print("overall clip fraction:", overall_clip_fraction)
        print("masked clip fraction:", masked_clip_fraction)

        return {
            "overall_clip_fraction": overall_clip_fraction,
            "masked_clip_fraction": masked_clip_fraction,
        }

    if mode == "3branch":
        stats = {}

        for output_name in ["Z_Output", "N_Output", "E_Output"]:
            y_comp = y[output_name]

            y_true = y_comp[..., 0:1]
            mask = y_comp[..., 1:2]

            stats[output_name] = compute_clip_stats(
                y_true,
                mask,
                output_name
            )

        return stats

    elif mode == "single_component":
        y_true = y[..., 0:1]
        mask = y[..., 1:2]

        return compute_clip_stats(
            y_true,
            mask,
            "single_component"
        )

    elif mode == "full":
        y_true = y[..., :3]
        mask = y[..., 3:]

        return compute_clip_stats(
            y_true,
            mask,
            "full"
        )

    else:
        raise ValueError("mode must be '3branch', 'single_component', or 'full'")

'------------------------------------------------------------------------------------------------------------------------------------------------------------------------'
# Ancillary VASA functions
class ExpandDims(tf.keras.layers.Layer):
    def __init__(self, axis=2, **kwargs):
        super().__init__(**kwargs)
        self.axis = axis
    def call(self, inputs):
        return tf.expand_dims(inputs, axis=self.axis)

class SqueezeAxis(tf.keras.layers.Layer):
    def __init__(self, axis=2, **kwargs):
        super().__init__(**kwargs)
        self.axis = axis

    def call(self, inputs):
        return tf.squeeze(inputs, axis=self.axis)

'------------------------------------------------------------------------------------------------------------------------------------------------------------------------'
# --- 1. The Custom Spatial Attention Layer (Corrected) ---
# This layer learns a bias directly from continuous distance values.
class SpatialAttention(MultiHeadAttention):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # A small MLP to learn a non-linear mapping from distance to bias
        # Input: distance (1) -> Hidden Layer (16) -> Output: bias (1)
        self.distance_mlp = tf.keras.Sequential([
            Dense(16, activation="relu"),
            Dense(
                1,
                kernel_initializer="zeros",
                bias_initializer="zeros"
            )
        ], name="distance_bias_mlp")

    # FIX: Implement the `call` method to accept the custom `distance_matrix` argument.
    def call(self, query, value, key=None, attention_mask=None, training=None, distance_matrix=None):
        if key is None:
            key = value

        # Standard linear projections for Q, K, V
        query = self._query_dense(query)
        key = self._key_dense(key)
        value = self._value_dense(value)

        # The core logic is now passed to your custom _compute_attention
        attention_output, attention_scores = self._compute_attention(
            query, key, value,
            attention_mask=attention_mask,
            training=training,
            distance_matrix=distance_matrix  # Pass the custom argument here
        )

        # Re-combine heads and apply final projection
        attention_output = self._output_dense(attention_output)
        return attention_output, attention_scores

    def _compute_attention(self, query, key, value, attention_mask=None, training=None, distance_matrix=None):
        if distance_matrix is None:
            raise ValueError("A `distance_matrix` must be provided to SpatialAttention.")

        # Standard attention score calculation
        query = tf.multiply(query, 1.0 / tf.math.sqrt(tf.cast(self._key_dim, self.dtype)))
        attention_scores = tf.einsum(self._dot_product_equation, key, query)
        
        # --- Generate and add the spatial bias ---
        # 1. Expand dims for the MLP: [B, S, S] -> [B, S, S, 1]
        expanded_distances = tf.expand_dims(distance_matrix, axis=-1)
        # 2. Compute bias: MLP maps [B, S, S, 1] -> [B, S, S, 1]
        distance_bias = self.distance_mlp(expanded_distances)
        # 3. Reshape for broadcasting: [B, S, S, 1] -> [B, 1, S, S]
        # This allows it to be added to scores of shape [B, H, S, S]
        reshaped_bias = tf.squeeze(distance_bias, axis=-1)[:, tf.newaxis, :, :]
        attention_scores += reshaped_bias
        # --- Bias addition complete ---
        
        if attention_mask is not None:
            attention_scores = self._masked_softmax(attention_scores, attention_mask)
        else:
            attention_scores = tf.nn.softmax(attention_scores, axis=-1)
        
        attention_scores_dropout = self._dropout_layer(attention_scores, training=training)
        attention_output = tf.einsum(self._combine_equation, attention_scores_dropout, value)
        
        return attention_output, attention_scores

# --- 2. The Spatially-Aware Transformer Block (No changes needed) ---
# This block uses our custom attention layer and is correct as is.
class SpatiallyAwareTransformerBlock(Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1, **kwargs):
        super().__init__(**kwargs)

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.rate = rate

        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads.")

        self.att = SpatialAttention(
            num_heads=num_heads,
            key_dim=embed_dim // num_heads,
            output_shape=embed_dim
        )

        self.ffn = tf.keras.Sequential([
            Dense(ff_dim, activation="relu"),
            Dense(embed_dim),
        ])

        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(rate)
        self.dropout2 = Dropout(rate)

    def call(self, inputs, distance_matrix, training=False):
        attn_output, _ = self.att(
            query=inputs,
            value=inputs,
            key=inputs,
            distance_matrix=distance_matrix,
            training=training
        )

        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)

        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)

        return self.layernorm2(out1 + ffn_output)

    def get_config(self):
        config = super().get_config()
        config.update({
            "embed_dim": self.embed_dim,
            "num_heads": self.num_heads,
            "ff_dim": self.ff_dim,
            "rate": self.rate,
        })
        return config

'------------------------------------------------------------------------------------------------------------------------------------------------------------------------'
# Loss functions
def masked_mse_loss(y_true_with_mask, y_pred, eps=1e-7):
    y_true = y_true_with_mask[..., :3]
    mask = y_true_with_mask[..., 3:]

    mse = tf.square(y_true - y_pred) * mask
    return tf.reduce_sum(mse) / (tf.reduce_sum(mask) + eps)

def masked_1c_mse_loss(y_true_with_mask, y_pred, eps=1e-8):
    y_true_z = y_true_with_mask[..., 0:1]
    mask_z = y_true_with_mask[..., 3:4]

    se = tf.square(y_true_z - y_pred) * mask_z

    return tf.reduce_sum(se) / (tf.reduce_sum(mask_z) + eps)

def masked_1c_mse_loss_2ch(y_true_with_mask, y_pred, eps=1e-8):
    y_true_z = y_true_with_mask[..., 0:1]
    mask_z = y_true_with_mask[..., 1:2]

    se = tf.square(y_true_z - y_pred) * mask_z

    return tf.reduce_sum(se) / (tf.reduce_sum(mask_z) + eps)

def masked_huber_loss(delta=1.0, eps=1e-8):
    """
    Masked Huber loss.

    y_true_with_mask: [B, S, T, 6]
        first 3 channels = y_true
        last 3 channels  = mask, 1 where loss applies

    y_pred: [B, S, T, 3]
    """

    def loss(y_true_with_mask, y_pred):
        y_true = y_true_with_mask[..., :3]
        mask = y_true_with_mask[..., 3:]

        error = y_true - y_pred
        abs_error = tf.abs(error)

        quadratic = tf.minimum(abs_error, delta)
        linear = abs_error - quadratic

        huber = 0.5 * tf.square(quadratic) + delta * linear

        huber = huber * mask

        return tf.reduce_sum(huber) / (tf.reduce_sum(mask) + eps)

    return loss

def masked_1c_huber_loss_2ch(delta=1.0, eps=1e-8):
    """
    Masked Huber loss for Z-only reconstruction.

    y_true_with_mask: [B, S, T, 2]
        channel 0 = Z target
        channel 1 = Z mask

    y_pred: [B, S, T, 1]
        predicted Z
    """

    def loss(y_true_with_mask, y_pred):
        y_true_z = y_true_with_mask[..., 0:1]
        mask_z = y_true_with_mask[..., 1:2]

        error = y_true_z - y_pred
        abs_error = tf.abs(error)

        quadratic = tf.minimum(abs_error, delta)
        linear = abs_error - quadratic

        huber = 0.5 * tf.square(quadratic) + delta * linear
        huber = huber * mask_z

        return tf.reduce_sum(huber) / (tf.reduce_sum(mask_z) + eps)

    return loss

'------------------------------------------------------------------------------------------------------------------------------------------------------------------------'
# Evaluate metrics separately during training
def masked_rmse_metric(y_true_with_mask, y_pred, eps=1e-7):
    y_true = y_true_with_mask[..., :3]
    mask = y_true_with_mask[..., 3:]

    mse = tf.square(y_true - y_pred) * mask
    mse = tf.reduce_sum(mse) / (tf.reduce_sum(mask) + eps)

    return tf.sqrt(mse + eps)

def masked_1c_rmse_metric(y_true_with_mask, y_pred, eps=1e-8):
    y_true_z = y_true_with_mask[..., 0:1]
    mask_z = y_true_with_mask[..., 3:4]

    se = tf.square(y_true_z - y_pred) * mask_z

    return tf.sqrt(tf.reduce_sum(se) / (tf.reduce_sum(mask_z) + eps))

def masked_1c_rmse_metric_2ch(y_true_with_mask, y_pred, eps=1e-8):
    """
    y_true_with_mask: [B, S, T, 2]
        channel 0 = Z target
        channel 1 = Z mask

    y_pred: [B, S, T, 1]
    """

    y_true_z = y_true_with_mask[..., 0:1]
    mask_z = y_true_with_mask[..., 1:2]

    se = tf.square(y_true_z - y_pred) * mask_z

    return tf.sqrt(tf.reduce_sum(se) / (tf.reduce_sum(mask_z) + eps))

def masked_corr_metric(y_true_with_mask, y_pred, eps=1e-8):
    """
    Masked zero-lag Pearson correlation.

    y_true_with_mask: [B, S, T, 6]
        first 3 channels = y_true
        last 3 channels  = mask, 1 where loss/metric applies

    y_pred: [B, S, T, 3]
    """

    y_true = y_true_with_mask[..., :3]
    mask = y_true_with_mask[..., 3:]

    y_true_masked = y_true * mask
    y_pred_masked = y_pred * mask

    count = tf.reduce_sum(mask) + eps

    true_mean = tf.reduce_sum(y_true_masked) / count
    pred_mean = tf.reduce_sum(y_pred_masked) / count

    true_centered = (y_true - true_mean) * mask
    pred_centered = (y_pred - pred_mean) * mask

    numerator = tf.reduce_sum(true_centered * pred_centered)

    denominator = tf.sqrt(
        tf.reduce_sum(tf.square(true_centered)) *
        tf.reduce_sum(tf.square(pred_centered)) +
        eps
    )

    return numerator / denominator

def masked_1c_corr_metric(y_true_with_mask, y_pred, eps=1e-8):
    """
    Zero-lag Pearson correlation for masked Z component only.

    y_true_with_mask: [B, S, T, 6]
        channel 0 = Z true
        channel 3 = Z mask

    y_pred: [B, S, T, 1]
        predicted Z
    """

    y_true_z = y_true_with_mask[..., 0:1]
    mask_z = y_true_with_mask[..., 3:4]

    y_true_masked = y_true_z * mask_z
    y_pred_masked = y_pred * mask_z

    count = tf.reduce_sum(mask_z) + eps

    true_mean = tf.reduce_sum(y_true_masked) / count
    pred_mean = tf.reduce_sum(y_pred_masked) / count

    true_centered = (y_true_z - true_mean) * mask_z
    pred_centered = (y_pred - pred_mean) * mask_z

    numerator = tf.reduce_sum(true_centered * pred_centered)

    denominator = tf.sqrt(
        tf.reduce_sum(tf.square(true_centered)) *
        tf.reduce_sum(tf.square(pred_centered)) +
        eps
    )

    return numerator / denominator

def masked_1c_corr_metric_2ch(y_true_with_mask, y_pred, eps=1e-8):
    """
    Zero-lag Pearson correlation for Z-only reconstruction.

    y_true_with_mask: [B, S, T, 2]
        channel 0 = Z target
        channel 1 = Z mask

    y_pred: [B, S, T, 1]
        predicted Z
    """

    y_true_z = y_true_with_mask[..., 0:1]
    mask_z = y_true_with_mask[..., 1:2]

    y_true_masked = y_true_z * mask_z
    y_pred_masked = y_pred * mask_z

    count = tf.reduce_sum(mask_z) + eps

    true_mean = tf.reduce_sum(y_true_masked) / count
    pred_mean = tf.reduce_sum(y_pred_masked) / count

    true_centered = (y_true_z - true_mean) * mask_z
    pred_centered = (y_pred - pred_mean) * mask_z

    numerator = tf.reduce_sum(true_centered * pred_centered)

    denominator = tf.sqrt(
        tf.reduce_sum(tf.square(true_centered)) *
        tf.reduce_sum(tf.square(pred_centered)) +
        eps
    )

    return numerator / denominator

'------------------------------------------------------------------------------------------------------------------------------------------------------------------------'
# Callbacks
class SumValCorrCallback(tf.keras.callbacks.Callback):
    def __init__(
        self,
        z_name="val_Z_Output_masked_1c_corr_metric_2ch",
        n_name="val_N_Output_masked_1c_corr_metric_2ch",
        e_name="val_E_Output_masked_1c_corr_metric_2ch",
        output_name="val_total_corr"
    ):
        super().__init__()
        self.z_name = z_name
        self.n_name = n_name
        self.e_name = e_name
        self.output_name = output_name

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}

        z_corr = logs.get(self.z_name)
        n_corr = logs.get(self.n_name)
        e_corr = logs.get(self.e_name)

        if z_corr is None or n_corr is None or e_corr is None:
            print(
                f"\nCould not compute {self.output_name}. "
                f"Available log keys: {list(logs.keys())}"
            )
            return

        logs[self.output_name] = z_corr + n_corr + e_corr

        print(
            f"\n{self.output_name}: {logs[self.output_name]:.5f} "
            f"(Z={z_corr:.5f}, N={n_corr:.5f}, E={e_corr:.5f})"
        )
'------------------------------------------------------------------------------------------------------------------------------------------------------------------------'
## -- Virtual Array Synthesis Architecture -- ##
## Concatenate encoders along sensor axis - input to Transformer now [30,150]
## U-Net-like skip connections per component branch

def temporal_pyramid_branch(input, prefix): 
    x_l = Conv2D(4,
                 kernel_size=(1,31),
                 strides=(1,1),
                 padding='same',
                 kernel_initializer='he_uniform',
                 name=f"{prefix}_Conv_1")(input)
    x_l = LayerNormalization(axis=-1, name=f"{prefix}_LN_2400")(x_l)
    x_l = activations.relu(x_l)

    skip_2400 = x_l  # [B, S, 2400, 4]

    x_l = Conv2D(8,
                 kernel_size=(1,21),
                 strides=(1,2),
                 padding='same',
                 kernel_initializer='he_uniform', 
                 name=f"{prefix}_Conv_2")(x_l)

    x_res = LayerNormalization(axis=-1)(x_l)
    x_res = activations.relu(x_res)
    x_res = Conv2D(8,
                   kernel_size=(1,15),
                   strides=(1,1), 
                   padding='same',
                   kernel_initializer='he_uniform', 
                   name=f"{prefix}_Conv_3a_Res")(x_res)
    x_res = LayerNormalization(axis=-1)(x_res)
    x_res = activations.relu(x_res)
    x_res = Conv2D(8,
                   kernel_size=(1,15),
                   strides=(1,1), 
                   padding='same',
                   kernel_initializer='he_uniform', 
                   name=f"{prefix}_Conv_3b_Res")(x_res)
    out = Add(name=f"{prefix}_ResAdd_1200")([x_l, x_res])
    x_l = LayerNormalization(axis=-1, name=f"{prefix}_LN_1200")(out)
    x_l = activations.relu(x_l)

    skip_1200 = x_l  # [B, S, 1200, 8]

    x_l = Conv2D(16,
                 kernel_size=(1,11),
                 strides=(1,2),
                 padding='same',
                 kernel_initializer='he_uniform', 
                 name=f"{prefix}_Conv_4")(x_l)
    x_l = LayerNormalization(axis=-1, name=f"{prefix}_LN_600")(x_l)
    x_l = activations.relu(x_l)

    skip_600 = x_l   # [B, S, 600, 16]

    x_l = Conv2D(16,
                 kernel_size=(1,7),
                 strides=(1,2),
                 padding='same',
                 kernel_initializer='he_uniform', 
                 name=f"{prefix}_Conv_5")(x_l)
    x_l = LayerNormalization(axis=-1, name=f"{prefix}_LN_300")(x_l)
    x_l = activations.relu(x_l)

    skip_300 = x_l   # [B, S, 300, 16]

    x_l = Conv2D(32,
                 kernel_size=(1,3),
                 strides=(1,2),
                 padding='same',
                 kernel_initializer='he_uniform', 
                 name=f"{prefix}_Conv_6")(x_l)
    x_l = LayerNormalization(axis=-1, name=f"{prefix}_LN_150")(x_l)
    x_l = activations.relu(x_l)

    return x_l, skip_2400, skip_1200, skip_600, skip_300


def decoder_branch(x, skips, sensor_dim, t_prime, name):
    # x: [B, S, 150]
    x_decoder = Reshape((sensor_dim, t_prime, 1), name=f"{name}_Input_Decoder")(x)

    # Lift Transformer tokens into a feature map
    x_decoder = Conv2D(
        64,
        (1, 1),
        padding="same",
        kernel_initializer="he_uniform",
        name=f"{name}_Decoder_Lift"
    )(x_decoder)
    x_decoder = LayerNormalization(axis=-1, name=f"{name}_Decoder_Lift_LN")(x_decoder)
    x_decoder = activations.gelu(x_decoder)

    # 150 skip
    skip_150 = Conv2D(
        64,
        (1, 1),
        padding="same",
        kernel_initializer="he_uniform",
        name=f"{name}_Skip150_Project"
    )(skips["skip_150"])

    x_decoder = Concatenate(axis=-1, name=f"{name}_Concat_Skip150")([x_decoder, skip_150])

    x_decoder = Conv2D(
        32,
        (3, 5),
        padding="same",
        kernel_initializer="he_uniform",
        name=f"{name}_Decoder_Conv_150"
    )(x_decoder)
    x_decoder = LayerNormalization(axis=-1, name=f"{name}_Decoder_LN_150")(x_decoder)
    x_decoder = activations.gelu(x_decoder)

    # -> 300
    x_decoder = UpSampling2D(size=(1, 2), interpolation="bilinear",
                             name=f"{name}_Upsample_1")(x_decoder)

    skip_300 = Conv2D(
        32,
        (1, 1),
        padding="same",
        kernel_initializer="he_uniform",
        name=f"{name}_Skip300_Project"
    )(skips["skip_300"])

    x_decoder = Concatenate(axis=-1, name=f"{name}_Concat_Skip300")([x_decoder, skip_300])

    x_decoder = Conv2D(
        32,
        (3, 5),
        padding="same",
        kernel_initializer="he_uniform",
        name=f"{name}_Decoder_Conv_300"
    )(x_decoder)
    x_decoder = LayerNormalization(axis=-1, name=f"{name}_Decoder_LN_300")(x_decoder)
    x_decoder = activations.gelu(x_decoder)

    # -> 600
    x_decoder = UpSampling2D(size=(1, 2), interpolation="bilinear",
                             name=f"{name}_Upsample_2")(x_decoder)

    skip_600 = Conv2D(
        32,
        (1, 1),
        padding="same",
        kernel_initializer="he_uniform",
        name=f"{name}_Skip600_Project"
    )(skips["skip_600"])

    x_decoder = Concatenate(axis=-1, name=f"{name}_Concat_Skip600")([x_decoder, skip_600])

    x_decoder = Conv2D(
        32,
        (3, 5),
        padding="same",
        kernel_initializer="he_uniform",
        name=f"{name}_Decoder_Conv_600"
    )(x_decoder)
    x_decoder = LayerNormalization(axis=-1, name=f"{name}_Decoder_LN_600")(x_decoder)
    x_decoder = activations.gelu(x_decoder)

    # -> 1200
    x_decoder = UpSampling2D(size=(1, 2), interpolation="bilinear",
                             name=f"{name}_Upsample_3")(x_decoder)

    skip_1200 = Conv2D(
        32,
        (1, 1),
        padding="same",
        kernel_initializer="he_uniform",
        name=f"{name}_Skip1200_Project"
    )(skips["skip_1200"])

    x_decoder = Concatenate(axis=-1, name=f"{name}_Concat_Skip1200")([x_decoder, skip_1200])

    x_decoder = Conv2D(
        32,
        (3, 3),
        padding="same",
        kernel_initializer="he_uniform",
        name=f"{name}_Decoder_Conv_1200"
    )(x_decoder)
    x_decoder = LayerNormalization(axis=-1, name=f"{name}_Decoder_LN_1200")(x_decoder)
    x_decoder = activations.gelu(x_decoder)

    # Residual block at 1200
    x_res = Conv2D(
        32,
        (3, 3),
        padding="same",
        kernel_initializer="he_uniform",
        name=f"{name}_Decoder_Conv_3a"
    )(x_decoder)
    x_res = LayerNormalization(axis=-1, name=f"{name}_Decoder_LN_3a")(x_res)
    x_res = activations.gelu(x_res)

    x_res = Conv2D(
        32,
        (3, 3),
        padding="same",
        kernel_initializer="he_uniform",
        name=f"{name}_Decoder_Conv_3b"
    )(x_res)

    x_decoder = Add(name=f"{name}_Decoder_Res_Add")([x_decoder, x_res])
    x_decoder = LayerNormalization(axis=-1, name=f"{name}_Decoder_LN_3b")(x_decoder)
    x_decoder = activations.gelu(x_decoder)

    # -> 2400
    x_decoder = UpSampling2D(size=(1, 2), interpolation="bilinear",
                             name=f"{name}_Upsample_4")(x_decoder)

    skip_2400 = Conv2D(
        32,
        (1, 1),
        padding="same",
        kernel_initializer="he_uniform",
        name=f"{name}_Skip2400_Project"
    )(skips["skip_2400"])

    x_decoder = Concatenate(axis=-1, name=f"{name}_Concat_Skip2400")([x_decoder, skip_2400])

    x_decoder = Conv2D(
        16,
        (3, 3),
        padding="same",
        kernel_initializer="he_uniform",
        name=f"{name}_Decoder_Conv_2400"
    )(x_decoder)
    x_decoder = LayerNormalization(axis=-1, name=f"{name}_Decoder_LN_2400")(x_decoder)
    x_decoder = activations.gelu(x_decoder)

    out = Conv2D(
        1,
        (3, 3),
        padding="same",
        activation="linear",
        name=f"{name}_Output"
    )(x_decoder)

    return out


def build_component_encoder(component_input, sensor_dim, temporal_dim, name):
    """
    component_input: [B, S, T, 2]
    returns:
        x_attn: [B, S, T']
        skips: dict of skip tensors
    """
    waveform = Lambda(lambda x: x[..., 0:1], name=f"{name}_Waveform_Input")(component_input)
    observed_mask = Lambda(lambda x: x[..., 1:2], name=f"{name}_ObservedMask_Input")(component_input)

    x, skip_2400, skip_1200, skip_600, skip_300 = temporal_pyramid_branch(waveform, prefix=name)

    x = Conv2D(128, (1, 1), padding="same", kernel_initializer="he_uniform",
               name=f"{name}_Expand")(x)
    x = LayerNormalization(axis=-1, name=f"{name}_Expand_LN")(x)
    x = activations.relu(x)

    sensor_observed = Lambda(
        lambda m: tf.reduce_max(m, axis=-1, keepdims=True),
        name=f"{name}_Sensor_Observed_Indicator"
    )(observed_mask)

    sensor_observed_ds = AveragePooling2D(
        pool_size=(1, temporal_dim // x.shape[2]),
        strides=(1, temporal_dim // x.shape[2]),
        padding="same",
        name=f"{name}_Sensor_Observed_Downsample"
    )(sensor_observed)

    x = Concatenate(axis=-1, name=f"{name}_Feature_With_ObservedMask")([x, sensor_observed_ds])

    x = Conv2D(64, (1, 1), padding="same", kernel_initializer="he_uniform",
               name=f"{name}_PostMask_Feature_Projection")(x)
    x = LayerNormalization(axis=-1, name=f"{name}_PostMask_LN")(x)
    x = activations.relu(x)

    skip_150 = x  # [B, S, 150, 64]

    # Learned feature pooling -> [B, S, T']
    q = Conv2D(16,
               kernel_size=(1, 3),
               activation="tanh",
               padding="same",
               kernel_initializer="he_uniform",
               name=f"{name}_FeatureAttn_Conv")(x)

    F = int(x.shape[-1])
    t_prime = int(x.shape[2])

    a = Dense(F, name=f"{name}_FeatureAttn_Dense")(q)
    a = Softmax(axis=-1, name=f"{name}_FeatureAttn_Score")(a)

    x_attn = Lambda(
        lambda t: tf.reduce_sum(t[0] * t[1], axis=-1),
        output_shape=(sensor_dim, t_prime),
        name=f"{name}_FeatureAttn_Pool"
    )([x, a])

    x_attn = LayerNormalization(
        epsilon=1e-6,
        name=f"{name}_FeatureAttn_LN"
    )(x_attn)

    skips = {
        "skip_2400": skip_2400,
        "skip_1200": skip_1200,
        "skip_600": skip_600,
        "skip_300": skip_300,
        "skip_150": skip_150,
    }

    return x_attn, skips


def build_30x30_distance_matrix(input_distance, sensor_dim):
    """
    input_distance: [B, 10, 10]
    output:         [B, 30, 30]
    token order = [Z stations 0..9, N stations 0..9, E stations 0..9]
    """
    station_ids = tf.tile(tf.range(sensor_dim), [3])

    def expand_dist(d):
        d = tf.gather(d, station_ids, axis=1)
        d = tf.gather(d, station_ids, axis=2)
        return d

    return Lambda(expand_dist, name="Expand_Distance_30x30")(input_distance)


def build_VASA(sensor_dim, temporal_dim, channel_dim, lr, num_heads=4, num_transformer_blocks=2):
    adam = Adam(learning_rate=lr, global_clipnorm=1.0)

    input_tensor_seismic = Input(
        shape=(sensor_dim, temporal_dim, channel_dim),
        name="seismic_input",
        dtype=tf.float32
    )
    input_distance = Input(shape=(sensor_dim, sensor_dim), name="distance_input")

    # Full input: [Z, N, E, obs_Z, obs_N, obs_E]
    z_input = Concatenate(axis=-1, name="Z_Branch_Input")([
        Lambda(lambda x: x[..., 0:1], name="Slice_Z_Wave")(input_tensor_seismic),
        Lambda(lambda x: x[..., 3:4], name="Slice_Z_Mask")(input_tensor_seismic),
    ])
    n_input = Concatenate(axis=-1, name="N_Branch_Input")([
        Lambda(lambda x: x[..., 1:2], name="Slice_N_Wave")(input_tensor_seismic),
        Lambda(lambda x: x[..., 4:5], name="Slice_N_Mask")(input_tensor_seismic),
    ])
    e_input = Concatenate(axis=-1, name="E_Branch_Input")([
        Lambda(lambda x: x[..., 2:3], name="Slice_E_Wave")(input_tensor_seismic),
        Lambda(lambda x: x[..., 5:6], name="Slice_E_Mask")(input_tensor_seismic),
    ])

    # Encode each component -> tokens [B, 10, 150] + skips
    z_tokens, z_skips = build_component_encoder(z_input, sensor_dim, temporal_dim, name="Z")
    n_tokens, n_skips = build_component_encoder(n_input, sensor_dim, temporal_dim, name="N")
    e_tokens, e_skips = build_component_encoder(e_input, sensor_dim, temporal_dim, name="E")

    t_prime = int(z_tokens.shape[-1])  # 150

    # Concatenate on token axis -> [B, 30, 150]
    x = Concatenate(axis=1, name="ComponentToken_Concat")([
        z_tokens, n_tokens, e_tokens
    ])

    token_distance = build_30x30_distance_matrix(input_distance, sensor_dim)

    station_ids = tf.tile(tf.range(sensor_dim), [3])
    component_ids = tf.repeat(tf.range(3), repeats=sensor_dim)

    station_emb = Embedding(
        input_dim=sensor_dim,
        output_dim=t_prime,
        name="Token_Station_Embedding"
    )(station_ids)

    component_emb = Embedding(
        input_dim=3,
        output_dim=t_prime,
        name="Token_Component_Embedding"
    )(component_ids)

    station_emb = Lambda(
        lambda t: tf.expand_dims(t, axis=0),
        name="Expand_Station_Embedding"
    )(station_emb)

    component_emb = Lambda(
        lambda t: tf.expand_dims(t, axis=0),
        name="Expand_Component_Embedding"
    )(component_emb)

    x = Add(name="Add_Token_Embeddings")([x, station_emb, component_emb])

    for i in range(num_transformer_blocks):
        x = SpatiallyAwareTransformerBlock(
            embed_dim=t_prime,
            num_heads=num_heads,
            ff_dim=t_prime * 2,
            rate=0.05,
            name=f"CrossComponent_Transformer_{i+1}"
        )(x, distance_matrix=token_distance)

    z_tokens_out = Lambda(
        lambda t: t[:, 0:sensor_dim, :],
        name="Z_Tokens_Out"
    )(x)

    n_tokens_out = Lambda(
        lambda t: t[:, sensor_dim:2 * sensor_dim, :],
        name="N_Tokens_Out"
    )(x)

    e_tokens_out = Lambda(
        lambda t: t[:, 2 * sensor_dim:3 * sensor_dim, :],
        name="E_Tokens_Out"
    )(x)

    z_out = decoder_branch(z_tokens_out, z_skips, sensor_dim, t_prime, name="Z")
    n_out = decoder_branch(n_tokens_out, n_skips, sensor_dim, t_prime, name="N")
    e_out = decoder_branch(e_tokens_out, e_skips, sensor_dim, t_prime, name="E")

    vasa = Model(
        inputs=[input_tensor_seismic, input_distance],
        outputs={
            "Z_Output": z_out,
            "N_Output": n_out,
            "E_Output": e_out,
        },
        name="VASA"
    )

    vasa.compile(
        optimizer=adam,
        loss={
            "Z_Output": masked_1c_mse_loss_2ch,
            "N_Output": masked_1c_mse_loss_2ch,
            "E_Output": masked_1c_mse_loss_2ch,
        },
        loss_weights={
            "Z_Output": 1.0,
            "N_Output": 1.0,
            "E_Output": 1.0,
        },
        metrics={
            "Z_Output": [masked_1c_rmse_metric_2ch, masked_1c_corr_metric_2ch],
            "N_Output": [masked_1c_rmse_metric_2ch, masked_1c_corr_metric_2ch],
            "E_Output": [masked_1c_rmse_metric_2ch, masked_1c_corr_metric_2ch],
        }
    )

    return vasa

'------------------------------------------------------------------------------------------------------------------------------------------------------------------------'
# Visualize and interpret results# Visualize and interpret results
def inverse_global_asinh_scale(x_scaled, global_center, global_scale, output_gain=1.0):
    global_center = np.asarray(global_center, dtype=np.float32)
    global_scale = np.asarray(global_scale, dtype=np.float32)

    return np.sinh(x_scaled / output_gain) * global_scale + global_center

def get_3branch_predictions(preds):
    """
    Handles either dict outputs or list outputs from Keras.
    Returns waveform predictions as [B, S, T, 3].
    Ignores MAG_Output if present.
    """
    if isinstance(preds, dict):
        z = preds["Z_Output"]
        n = preds["N_Output"]
        e = preds["E_Output"]
    elif isinstance(preds, list):
        z, n, e = preds[:3]
    else:
        raise ValueError("Expected model predictions to be dict or list.")

    return np.concatenate([z, n, e], axis=-1)

def get_mag_prediction(preds):
    """
    Extract scalar magnitude prediction from model output if present.
    Returns None if MAG_Output is not available.
    """
    if isinstance(preds, dict):
        mag_pred = preds.get("MAG_Output", None)
        if mag_pred is None:
            return None
        return float(np.asarray(mag_pred)[0, 0])

    if isinstance(preds, list) and len(preds) >= 4:
        return float(np.asarray(preds[3])[0, 0])

    return None

def extract_cross_component_attention_summary(
    model,
    seismic_input,
    distance_input,
    station_order=None,
    batch_size=32,
    focus="masked_sensor",
    include_self=True,
):
    """
    Summarize transformer attention over the 30 fused sensor-component tokens.

    Parameters
    ----------
    model : tf.keras.Model
        Trained VASA model built by build_VASA(...).
    seismic_input : array_like, shape [N, S, T, 6]
        Model-ready seismic input. Last 3 channels must be observed-mask
        channels exactly as used during training/inference.
    distance_input : array_like, shape [N, S, S] or [S, S]
        Model-ready distance input. A single [S, S] matrix will be broadcast
        across all examples.
    station_order : list[str] or None
        Optional station names. If None, uses S0, S1, ...
    batch_size : int
        Batch size used while extracting attention.
    focus : {"masked_sensor", "all_queries"}
        How to aggregate query tokens.
        - "masked_sensor": average only over the 3 query tokens belonging to
          the masked sensor in each sample. This is usually the most useful
          view for single-sensor reconstruction.
        - "all_queries": average over all 30 query tokens.
    include_self : bool
        Whether to keep attention directed to the masked sensor's own 3 tokens
        when focus="masked_sensor".

    Returns
    -------
    results : dict
        Contains:
        - "token_summary": DataFrame with 30 rows (one per sensor-component token)
        - "sensor_summary": DataFrame with 10 rows (attention summed over Z/N/E)
        - "attention_heatmap": ndarray [30, 30]
        - "attention_heatmap_df": DataFrame [30, 30]
        - "per_block_attention_heatmaps": dict of ndarray [30, 30]
        - "per_block_token_summary": dict of per-block DataFrames
        - "block_names": list of transformer block names
        - "token_labels": list of 30 token labels
    """
    seismic_input = np.asarray(seismic_input, dtype=np.float32)
    distance_input = np.asarray(distance_input, dtype=np.float32)

    if seismic_input.ndim != 4:
        raise ValueError("seismic_input must have shape [N, S, T, 6].")
    if seismic_input.shape[-1] != 6:
        raise ValueError("seismic_input last dimension must be 6: [Z,N,E,obs_Z,obs_N,obs_E].")

    n_examples, sensor_dim, _, _ = seismic_input.shape
    if distance_input.ndim == 2:
        distance_input = np.broadcast_to(distance_input, (n_examples,) + distance_input.shape).copy()
    elif distance_input.ndim != 3:
        raise ValueError("distance_input must have shape [N, S, S] or [S, S].")

    if distance_input.shape[0] != n_examples:
        raise ValueError("distance_input batch dimension must match seismic_input.")
    if distance_input.shape[1:] != (sensor_dim, sensor_dim):
        raise ValueError("distance_input spatial dimensions must match sensor_dim.")

    if station_order is None:
        station_order = [f"S{i}" for i in range(sensor_dim)]
    if len(station_order) != sensor_dim:
        raise ValueError("station_order length must equal sensor_dim.")

    block_layers = [
        layer for layer in model.layers
        if isinstance(layer, SpatiallyAwareTransformerBlock)
    ]
    if len(block_layers) == 0:
        raise ValueError("No SpatiallyAwareTransformerBlock layers found in model.")

    pre_block_tensors = [model.get_layer("Add_Token_Embeddings").output]
    for block in block_layers[:-1]:
        pre_block_tensors.append(block.output)

    inspector = Model(
        inputs=model.inputs,
        outputs=pre_block_tensors + [model.get_layer("Expand_Distance_30x30").output],
        name="AttentionInspector"
    )

    component_names = ["Z", "N", "E"]
    token_station = station_order * 3
    token_component = sum(([comp] * sensor_dim for comp in component_names), [])
    token_labels = [
        f"{sta}_{comp}" for comp in component_names for sta in station_order
    ]

    block_sums = {
        block.name: np.zeros(3 * sensor_dim, dtype=np.float64)
        for block in block_layers
    }
    block_matrix_sums = {
        block.name: np.zeros((3 * sensor_dim, 3 * sensor_dim), dtype=np.float64)
        for block in block_layers
    }
    row_counts = np.zeros(3 * sensor_dim, dtype=np.float64)
    total_examples = 0

    for start in range(0, n_examples, batch_size):
        end = min(start + batch_size, n_examples)
        x_batch = seismic_input[start:end]
        d_batch = distance_input[start:end]

        inspector_outputs = inspector(
            {
                "seismic_input": x_batch,
                "distance_input": d_batch,
            },
            training=False
        )

        if not isinstance(inspector_outputs, (list, tuple)):
            inspector_outputs = [inspector_outputs]

        token_distance = inspector_outputs[-1]
        block_inputs = inspector_outputs[:-1]

        obs_mask = x_batch[..., 3:6]
        obs_per_sensor = obs_mask.mean(axis=(2, 3))
        masked_sensor_idx = np.argmin(obs_per_sensor, axis=1)

        for block, block_input in zip(block_layers, block_inputs):
            _, scores = block.att(
                query=block_input,
                value=block_input,
                key=block_input,
                distance_matrix=token_distance,
                training=False
            )
            scores_np = scores.numpy()

            for bi in range(scores_np.shape[0]):
                if focus == "masked_sensor":
                    sidx = int(masked_sensor_idx[bi])
                    query_idx = np.array([sidx, sensor_dim + sidx, 2 * sensor_dim + sidx], dtype=int)
                    sample_scores = np.take(scores_np[bi], query_idx, axis=1)
                    if not include_self:
                        sample_scores = sample_scores.copy()
                        sample_scores[:, :, query_idx] = np.nan
                    sample_matrix = np.nanmean(sample_scores, axis=0)
                    sample_importance = np.nanmean(sample_scores, axis=(0, 1))
                    block_matrix_sums[block.name][query_idx, :] += sample_matrix
                    row_counts[query_idx] += 1
                elif focus == "all_queries":
                    sample_matrix = scores_np[bi].mean(axis=0)
                    sample_importance = sample_matrix.mean(axis=0)
                    block_matrix_sums[block.name] += sample_matrix
                    row_counts += 1
                else:
                    raise ValueError("focus must be 'masked_sensor' or 'all_queries'.")

                block_sums[block.name] += sample_importance
                total_examples += 1 if block.name == block_layers[0].name else 0

    if total_examples == 0:
        raise ValueError("No examples were processed.")
    if np.any(row_counts == 0):
        missing_rows = np.where(row_counts == 0)[0]
        warnings.warn(
            f"Some attention heatmap rows were never populated: {missing_rows.tolist()}. "
            "This usually means those sensor/component queries were not represented in the provided inputs."
        )

    per_block_token_summary = {}
    per_block_attention_heatmaps = {}
    for block in block_layers:
        mean_attention = block_sums[block.name] / total_examples
        df_block = pd.DataFrame({
            "token_index": np.arange(3 * sensor_dim),
            "station": token_station,
            "component": token_component,
            "token_label": token_labels,
            "mean_attention": mean_attention,
        }).sort_values("mean_attention", ascending=False, ignore_index=True)
        per_block_token_summary[block.name] = df_block

        heatmap_block = block_matrix_sums[block.name].copy()
        valid_rows = row_counts > 0
        heatmap_block[valid_rows] = heatmap_block[valid_rows] / row_counts[valid_rows, None]
        per_block_attention_heatmaps[block.name] = heatmap_block

    stacked = np.stack(
        [block_sums[block.name] / total_examples for block in block_layers],
        axis=0
    )
    mean_attention_all_blocks = stacked.mean(axis=0)

    stacked_heatmaps = np.stack(
        [per_block_attention_heatmaps[block.name] for block in block_layers],
        axis=0
    )
    attention_heatmap = stacked_heatmaps.mean(axis=0)
    attention_heatmap_df = pd.DataFrame(
        attention_heatmap,
        index=token_labels,
        columns=token_labels
    )

    token_summary = pd.DataFrame({
        "token_index": np.arange(3 * sensor_dim),
        "station": token_station,
        "component": token_component,
        "token_label": token_labels,
        "mean_attention": mean_attention_all_blocks,
    }).sort_values("mean_attention", ascending=False, ignore_index=True)

    sensor_attention = []
    for si, sta in enumerate(station_order):
        idxs = [si, sensor_dim + si, 2 * sensor_dim + si]
        sensor_attention.append(mean_attention_all_blocks[idxs].sum())

    sensor_summary = pd.DataFrame({
        "sensor_index": np.arange(sensor_dim),
        "station": station_order,
        "mean_attention_sum": sensor_attention,
    }).sort_values("mean_attention_sum", ascending=False, ignore_index=True)

    return {
        "token_summary": token_summary,
        "sensor_summary": sensor_summary,
        "attention_heatmap": attention_heatmap,
        "attention_heatmap_df": attention_heatmap_df,
        "per_block_attention_heatmaps": per_block_attention_heatmaps,
        "per_block_token_summary": per_block_token_summary,
        "block_names": [block.name for block in block_layers],
        "token_labels": token_labels,
        "row_counts": row_counts,
    }

def plot_cross_component_attention_heatmap(
    attention_results,
    figsize=(14, 12),
    cmap="magma",
    vmin=None,
    vmax=None,
    annotate=False,
    fmt=".2f",
    show_component_dividers=True,
    title=None,
):
    """
    Plot a 30x30 cross-component attention heatmap returned by
    extract_cross_component_attention_summary(...).

    Parameters
    ----------
    attention_results : dict
        Output of extract_cross_component_attention_summary(...).
    figsize : tuple
        Figure size in inches.
    cmap : str
        Matplotlib colormap.
    vmin, vmax : float or None
        Optional color limits.
    annotate : bool
        Whether to write values into each heatmap cell.
    fmt : str
        Annotation format if annotate=True.
    show_component_dividers : bool
        Whether to draw separators between Z, N, and E token blocks.
    title : str or None
        Optional plot title.

    Returns
    -------
    fig, ax, heatmap_df
    """
    heatmap_df = attention_results["attention_heatmap_df"]
    token_labels = attention_results["token_labels"]
    matrix = heatmap_df.values
    n_tokens = matrix.shape[0]

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(matrix, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto", origin="upper")

    ax.set_xticks(np.arange(n_tokens))
    ax.set_yticks(np.arange(n_tokens))
    ax.set_xticklabels(token_labels, rotation=90, fontsize=8)
    ax.set_yticklabels(token_labels, fontsize=8)
    ax.set_xlabel("Key token")
    ax.set_ylabel("Query token")

    if title is None:
        title = "Cross-Component Attention Heatmap"
    ax.set_title(title)

    if show_component_dividers and n_tokens % 3 == 0:
        sensor_dim = n_tokens // 3
        for boundary in [sensor_dim - 0.5, 2 * sensor_dim - 0.5]:
            ax.axhline(boundary, color="white", linewidth=1.2, alpha=0.8)
            ax.axvline(boundary, color="white", linewidth=1.2, alpha=0.8)

    if annotate:
        for i in range(n_tokens):
            for j in range(n_tokens):
                ax.text(
                    j, i, format(matrix[i, j], fmt),
                    ha="center", va="center",
                    fontsize=6, color="white"
                )

    cbar = fig.colorbar(im, ax=ax, pad=0.02)
    cbar.set_label("Mean attention")
    fig.tight_layout()

    return fig, ax, heatmap_df

def norm_xcorr(t_data, data1, data2, eps=1e-12):
    a = np.asarray(data1, dtype=np.float64)
    b = np.asarray(data2, dtype=np.float64)

    a_std = np.std(a)
    b_std = np.std(b)

    if a_std < eps or b_std < eps:
        lags_sec = np.hstack((np.flipud(-t_data)[0:len(t_data)-1], t_data))
        x_corr = np.zeros_like(lags_sec, dtype=np.float64)
        return lags_sec, x_corr

    a = (a - np.mean(a)) / (a_std * len(a))
    b = (b - np.mean(b)) / b_std

    x_corr = np.correlate(a, b, mode="full")
    lags_sec = np.hstack((np.flipud(-t_data)[0:len(t_data)-1], t_data))

    return lags_sec, x_corr


def raw_amplitude_ratios(y_true, y_pred, eps=1e-12):
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)

    true_rms = np.sqrt(np.mean(y_true ** 2)) + eps
    pred_rms = np.sqrt(np.mean(y_pred ** 2)) + eps
    rms_ratio = pred_rms / true_rms

    true_peak = np.max(np.abs(y_true)) + eps
    pred_peak = np.max(np.abs(y_pred))
    peak_ratio = pred_peak / true_peak

    return {
        "rms_ratio": float(rms_ratio),
        "peak_ratio": float(peak_ratio),
    }

def plot_random_test_reconstruction(
    model,
    X_test,
    D_norm,
    global_center,
    global_scale,
    meta_test=None,
    station_order=None,
    event_index=None,
    sensor_index=None,
    seed=None,
    output_gain=1.0,
    final_clip=100.0,
    plot_inverse=True,
    show_context_sensors=False,
    sampling_rate=40.0,
    save_fig=False,
    save_path=None,
    dpi=300,
    xlim=(0, 60),
    legend_locs=None,
    metric_text_locs=None,
):
    rng = np.random.default_rng(seed)

    n_events, sensor_dim, temporal_dim, channel_dim = X_test.shape

    if station_order is None:
        station_order = [f"S{i}" for i in range(sensor_dim)]

    if event_index is None:
        event_index = rng.integers(0, n_events)

    if sensor_index is None:
        sensor_index = rng.integers(0, sensor_dim)

    x_clean = X_test[event_index].astype(np.float32)

    event_info = None
    title_line_1 = f"Event {event_index}"

    if meta_test is not None:
        event_info = meta_test.iloc[event_index]

        evid = event_info.get("EVID", event_info.get("ev_id", "NA"))
        date_str = event_info.get("YY/MM/DD", event_info.get("YYY/MM/DD", "NA"))
        mag = event_info.get("MAG", "NA")
        dist_km = event_info.get("distance_km", "NA")

        mag_str = f"{float(mag):.2f}" if pd.notna(mag) else "NA"
        dist_str = f"{float(dist_km):.1f} km" if pd.notna(dist_km) else "NA"

        title_line_1 = f"EVID: {evid} | Date: {date_str} | Dist: {dist_str} | M: {mag_str}"

    x_scaled = global_asinh_scale(
        tf.constant(x_clean),
        global_center=tf.constant(global_center, dtype=tf.float32),
        global_scale=tf.constant(global_scale, dtype=tf.float32),
        output_gain=output_gain,
        final_clip=final_clip
    ).numpy()

    mask = np.zeros_like(x_scaled, dtype=np.float32)
    mask[sensor_index, :, :] = 1.0

    observed = 1.0 - mask
    x_masked = x_scaled * observed

    x_model = np.concatenate([x_masked, observed], axis=-1)[None, ...]
    d_model = D_norm.astype(np.float32)[None, ...]

    preds = model.predict(
        {
            "seismic_input": x_model,
            "distance_input": d_model
        },
        verbose=0
    )

    y_pred_scaled = get_3branch_predictions(preds)[0]

    true_masked = x_scaled[sensor_index]
    pred_masked = y_pred_scaled[sensor_index]

    observed_scaled = x_scaled.copy()
    observed_scaled[sensor_index] = np.nan

    if plot_inverse:
        true_plot = inverse_global_asinh_scale(
            true_masked,
            global_center,
            global_scale,
            output_gain
        )
        pred_plot = inverse_global_asinh_scale(
            pred_masked,
            global_center,
            global_scale,
            output_gain
        )
        observed_plot = inverse_global_asinh_scale(
            observed_scaled,
            global_center,
            global_scale,
            output_gain
        )
        ylabel = "Vel. [km/s]"
        title_suffix = "0.5 - 5 Hz"
    else:
        true_plot = true_masked
        pred_plot = pred_masked
        observed_plot = observed_scaled
        ylabel = "Scaled amplitude"
        title_suffix = "scaled model space"

    component_names = ["BHZ", "BHN", "BHE"]
    time = np.arange(temporal_dim) / sampling_rate

    def _resolve_per_subplot_setting(setting, ci, comp_name, default=None):
        """
        Resolve per-subplot settings.

        Supported:
        - None -> default
        - scalar/str/tuple -> same setting for all subplots
        - list/tuple length 3 -> per-subplot
        - dict with keys component names or indices
        """
        if setting is None:
            return default

        if isinstance(setting, dict):
            if comp_name in setting:
                return setting[comp_name]
            if ci in setting:
                return setting[ci]
            return default

        if isinstance(setting, (list, tuple)):
            # Treat 2-tuples like a single coordinate setting for all subplots
            if len(setting) == 2 and not isinstance(setting[0], (list, tuple, dict)):
                return setting
            if len(setting) != 3:
                raise ValueError(
                    "Per-subplot list/tuple settings must have length 3."
                )
            return setting[ci]

        return setting

    def _resolve_legend_loc(ci, comp_name):
        return _resolve_per_subplot_setting(
            legend_locs,
            ci,
            comp_name,
            default="upper right" if ci == 0 else None
        )

    def _resolve_metric_text_loc(ci, comp_name):
        """
        Returns a dict with:
        {
            "x": float,
            "y": float,
            "ha": str,
            "va": str,
        }
        """
        default = {"x": 0.02, "y": 0.95, "ha": "left", "va": "top"}
        loc = _resolve_per_subplot_setting(metric_text_locs, ci, comp_name, default=default)

        # Named presets
        if isinstance(loc, str):
            presets = {
                "upper left":  {"x": 0.02, "y": 0.95, "ha": "left",  "va": "top"},
                "upper right": {"x": 0.98, "y": 0.95, "ha": "right", "va": "top"},
                "lower left":  {"x": 0.02, "y": 0.05, "ha": "left",  "va": "bottom"},
                "lower right": {"x": 0.98, "y": 0.05, "ha": "right", "va": "bottom"},
                "center left": {"x": 0.02, "y": 0.50, "ha": "left",  "va": "center"},
                "center right":{"x": 0.98, "y": 0.50, "ha": "right", "va": "center"},
                "center":      {"x": 0.50, "y": 0.50, "ha": "center","va": "center"},
            }
            if loc not in presets:
                raise ValueError(
                    "metric_text_locs preset must be one of: "
                    f"{list(presets.keys())}"
                )
            return presets[loc]

        # Dict form
        if isinstance(loc, dict):
            merged = default.copy()
            merged.update(loc)
            return merged

        # Tuple/list coordinate form
        if isinstance(loc, (tuple, list)) and len(loc) == 2:
            x, y = loc
            ha = "left" if x <= 0.5 else "right"
            va = "bottom" if y <= 0.5 else "top"
            return {"x": x, "y": y, "ha": ha, "va": va}

        raise ValueError(
            "metric_text_locs must be None, a preset string, a 2-tuple/list, "
            "a dict, a length-3 list/tuple, or a dict keyed by component/index."
        )

    fig, axes = plt.subplots(
        3,
        1,
        figsize=(14, 8),
        sharex=True,
        clear=True
    )

    xcorr_stats = {}
    amplitude_stats = {}

    for ci, ax in enumerate(axes):
        context_labeled = False

        if show_context_sensors:
            for si in range(sensor_dim):
                if si == sensor_index:
                    continue

                ax.plot(
                    time,
                    observed_plot[si, :, ci],
                    color="0.75",
                    linewidth=0.7,
                    alpha=0.55,
                    label="Observed context" if not context_labeled else None
                )
                if not context_labeled:
                    context_labeled = True

        ax.plot(
            time,
            true_plot[:, ci],
            color="black",
            linewidth=1.8,
            label=f"True masked {station_order[sensor_index]}"
        )

        ax.plot(
            time,
            pred_plot[:, ci],
            color="crimson",
            linewidth=1.4,
            linestyle="--",
            label="Prediction"
        )

        lags_sec, x_corr = norm_xcorr(
            time,
            true_plot[:, ci],
            pred_plot[:, ci]
        )
        best_idx = int(np.argmax(x_corr))
        max_corr = float(x_corr[best_idx])
        best_lag_sec = float(lags_sec[best_idx])

        xcorr_stats[component_names[ci]] = {
            "max_corr": max_corr,
            "best_lag_sec": best_lag_sec,
        }

        ratios = raw_amplitude_ratios(
            true_plot[:, ci],
            pred_plot[:, ci]
        )

        amplitude_stats[component_names[ci]] = ratios

        metric_loc = _resolve_metric_text_loc(ci, component_names[ci])

        ax.text(
            metric_loc["x"],
            metric_loc["y"],
            (
                f"max xcorr = {max_corr:.3f}\n"
                f"best lag = {best_lag_sec:.4f} s\n"
                f"RMS ratio = {ratios['rms_ratio']:.3f}\n"
                f"Peak ratio = {ratios['peak_ratio']:.3f}"
            ),
            transform=ax.transAxes,
            va=metric_loc["va"],
            ha=metric_loc["ha"],
            fontsize=10,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.85, edgecolor="0.7")
        )

        ax.set_ylabel(f"{component_names[ci]}\n{ylabel}")
        ax.grid(alpha=0.25)
        ax.set_xlim(xlim)

        if ci == 0:
            ax.set_title(
                f"{title_line_1}\n"
                f"Masked sensor: {station_order[sensor_index]} | {title_suffix}"
            )

        legend_loc = _resolve_legend_loc(ci, component_names[ci])
        if legend_loc is not None:
            ax.legend(loc=legend_loc)

    axes[-1].set_xlabel("Time (s)")
    fig.tight_layout()

    if save_fig:
        if save_path is None:
            save_path = (
                f"reconstruction_event_{event_index}_"
                f"sensor_{station_order[sensor_index]}.png"
            )

        fig.savefig(
            save_path,
            dpi=dpi,
            bbox_inches="tight"
        )

        print(f"Saved figure: {save_path}")

    plt.show()

    return {
        "event_index": event_index,
        "sensor_index": sensor_index,
        "station": station_order[sensor_index],
        "event_info": event_info,
        "x_model": x_model,
        "prediction_scaled": y_pred_scaled,
        "true_scaled": x_scaled,
        "mask": mask,
        "figure": fig,
        "axes": axes,
        "xcorr_stats": xcorr_stats,
        "amplitude_stats": amplitude_stats,
        "save_path": save_path if save_fig else None,
    }


def plot_component_all_sensors_single_reconstruction(
    model,
    X_data,
    D_norm,
    global_center,
    global_scale,
    component="Z",
    meta_df=None,
    station_order=None,
    event_index=None,
    sensor_index=None,
    seed=None,
    output_gain=1.0,
    final_clip=100.0,
    plot_inverse=True,
    sampling_rate=40.0,
    normalize_each_trace=False,
    save_fig=False,
    save_path=None,
    dpi=300,
    xlim=(0, 60)
):
    rng = np.random.default_rng(seed)

    n_events, sensor_dim, temporal_dim, _ = X_data.shape

    if station_order is None:
        station_order = [f"S{i}" for i in range(sensor_dim)]

    if event_index is None:
        event_index = rng.integers(0, n_events)

    if sensor_index is None:
        sensor_index = rng.integers(0, sensor_dim)

    component = component.upper()
    comp_to_idx = {"Z": 0, "N": 1, "E": 2, "BHZ": 0, "BHN": 1, "BHE": 2}
    if component not in comp_to_idx:
        raise ValueError("component must be one of 'Z', 'N', 'E', 'BHZ', 'BHN', 'BHE'")

    ci = comp_to_idx[component]
    comp_name = ["BHZ", "BHN", "BHE"][ci]

    x_clean = X_data[event_index].astype(np.float32)

    event_info = None
    title_line_1 = f"Event {event_index}"

    if meta_df is not None:
        event_info = meta_df.iloc[event_index]
        evid = event_info.get("EVID", event_info.get("ev_id", "NA"))
        date_str = event_info.get("YY/MM/DD", event_info.get("YYY/MM/DD", "NA"))
        mag = event_info.get("MAG", "NA")
        dist_km = event_info.get("distance_km", "NA")

        mag_str = f"{float(mag):.2f}" if pd.notna(mag) else "NA"
        dist_str = f"{float(dist_km):.1f} km" if pd.notna(dist_km) else "NA"

        title_line_1 = f"EVID: {evid} | Date: {date_str} | Dist: {dist_str} | M: {mag_str}"

    x_scaled = global_asinh_scale(
        tf.constant(x_clean),
        global_center=tf.constant(global_center, dtype=tf.float32),
        global_scale=tf.constant(global_scale, dtype=tf.float32),
        output_gain=output_gain,
        final_clip=final_clip
    ).numpy()

    mask = np.zeros_like(x_scaled, dtype=np.float32)
    mask[sensor_index, :, :] = 1.0

    observed = 1.0 - mask
    x_masked = x_scaled * observed

    x_model = np.concatenate([x_masked, observed], axis=-1)[None, ...]
    d_model = D_norm.astype(np.float32)[None, ...]

    preds = model.predict(
        {
            "seismic_input": x_model,
            "distance_input": d_model
        },
        verbose=0
    )

    y_pred_scaled = get_3branch_predictions(preds)[0]

    if plot_inverse:
        true_plot = inverse_global_asinh_scale(
            x_scaled,
            global_center,
            global_scale,
            output_gain
        )
        pred_plot = inverse_global_asinh_scale(
            y_pred_scaled,
            global_center,
            global_scale,
            output_gain
        )
        amp_label = "Velocity amplitude [km/s]"
        title_suffix = "0.5 - 5 Hz"
    else:
        true_plot = x_scaled
        pred_plot = y_pred_scaled
        amp_label = "Scaled amplitude"
        title_suffix = "scaled model space"

    true_comp = true_plot[:, :, ci]
    pred_comp = pred_plot[:, :, ci]

    if normalize_each_trace:
        def _norm_rows(x, eps=1e-12):
            scale = np.max(np.abs(x), axis=1, keepdims=True) + eps
            return x / scale
        true_comp = _norm_rows(true_comp)
        pred_comp = _norm_rows(pred_comp)

    global_max = max(np.max(np.abs(true_comp)), np.max(np.abs(pred_comp)), 1e-12)
    spacing = 2.0 * global_max

    time = np.arange(temporal_dim) / sampling_rate
    offsets = np.arange(sensor_dim)[::-1] * spacing

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(14, 10),
        sharex=True,
        clear=True
    )

    # Top: true/observed traces for all sensors
    ax = axes[0]
    for si in range(sensor_dim):
        ax.plot(
            time,
            true_comp[si] + offsets[si],
            color="black",
            linewidth=2.0 if si == sensor_index else 1.2,
            alpha=1.0 if si == sensor_index else 0.85
        )

    ax.set_title(
        f"{title_line_1}\n"
        f"Observed {comp_name} | Masked sensor: {station_order[sensor_index]} | {title_suffix}"
    )
    ax.set_ylabel("Station (stacked)")
    ax.set_yticks(offsets)
    ax.set_yticklabels(station_order)
    ax.grid(alpha=0.25)
    ax.set_xlim(xlim)

    # Bottom: all sensors in black context, only masked sensor prediction in red
    ax = axes[1]
    for si in range(sensor_dim):
        if si == sensor_index:
            continue
        ax.plot(
            time,
            true_comp[si] + offsets[si],
            color="black",
            linewidth=1.0,
            alpha=0.8
        )

    ax.plot(
        time,
        pred_comp[sensor_index] + offsets[sensor_index],
        color="crimson",
        linewidth=2.0,
        label=f"Predicted {station_order[sensor_index]}"
    )

    ax.set_title(f"Predicted {comp_name} for masked sensor only")
    ax.set_ylabel("Station (stacked)")
    ax.set_xlabel(f"Time (s)\nTrace amplitudes are in {amp_label}, vertically offset for display")
    ax.set_yticks(offsets)
    ax.set_yticklabels(station_order)
    ax.grid(alpha=0.25)
    ax.legend(loc="upper right")

    fig.tight_layout()

    if save_fig:
        if save_path is None:
            save_path = (
                f"single_sensor_reconstruction_{comp_name}_"
                f"event_{event_index}_masked_{station_order[sensor_index]}.png"
            )

        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        print(f"Saved figure: {save_path}")

    plt.show()

    return {
        "event_index": event_index,
        "sensor_index": sensor_index,
        "component": comp_name,
        "station": station_order[sensor_index],
        "event_info": event_info,
        "true_component_array": true_comp,
        "pred_component_array": pred_comp,
        "masked_sensor_true": true_comp[sensor_index],
        "masked_sensor_pred": pred_comp[sensor_index],
        "offsets": offsets,
        "figure": fig,
        "axes": axes,
        "save_path": save_path if save_fig else None,
    }

def load_site_table(site_file, elev_in_km=True):
    """
    Read a simple site file with columns:
        STATION LAT LON ELEV

    Returns:
        dict like:
        {
            "BPH01": {"lat": 33.611000, "lon": -116.455498, "elev_m": 1292.0},
            ...
        }
    """
    site_file = Path(site_file)
    site_dict = {}

    with open(site_file, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            parts = line.split()
            if len(parts) < 4:
                continue

            sta = parts[0]
            lat = float(parts[1])
            lon = float(parts[2])
            elev = float(parts[3])

            elev_m = elev * 1000.0 if elev_in_km else elev

            site_dict[sta] = {
                "lat": lat,
                "lon": lon,
                "elev_m": elev_m,
            }

    return site_dict

def build_obspy_streams_for_reconstruction(
    model,
    X_data,
    D_norm,
    global_center,
    global_scale,
    station_order,
    site_file,
    component="Z",              # "Z", "N", "E", "BHZ", "BHN", "BHE"
    event_index=0,
    sensor_index=0,             # masked / reconstructed sensor
    meta_df=None,               # <- added
    output_gain=1.0,
    final_clip=100.0,
    plot_inverse=True,
    sampling_rate=40.0,
    starttime=None,
    network="XX",
    location="",
    elev_in_km=True,
):
    """
    Return two ObsPy Streams:

    1. st_observed:
       all observed/true waveforms for the selected component

    2. st_reconstructed:
       same as st_observed, except the selected sensor is replaced by the
       model prediction for that component

    SAC headers added:
        stla, stlo, stel

    If starttime is None and meta_df is provided, uses:
        meta_df.iloc[event_index]["window_start_utc"]
    """
    site_dict = load_site_table(site_file, elev_in_km=elev_in_km)

    component = component.upper()
    comp_to_idx = {"Z": 0, "N": 1, "E": 2, "BHZ": 0, "BHN": 1, "BHE": 2}
    comp_to_chan = {0: "BHZ", 1: "BHN", 2: "BHE"}

    if component not in comp_to_idx:
        raise ValueError("component must be one of 'Z', 'N', 'E', 'BHZ', 'BHN', 'BHE'")

    ci = comp_to_idx[component]
    chan = comp_to_chan[ci]

    # Determine stream start time
    if starttime is None:
        if meta_df is not None:
            window_start_utc = meta_df.iloc[event_index]["window_start_utc"]
            starttime = UTCDateTime(str(window_start_utc))
        else:
            starttime = UTCDateTime(0)
    elif not isinstance(starttime, UTCDateTime):
        starttime = UTCDateTime(starttime)

    x_clean = X_data[event_index].astype(np.float32)   # [S, T, 3]

    x_scaled = global_asinh_scale(
        tf.constant(x_clean),
        global_center=tf.constant(global_center, dtype=tf.float32),
        global_scale=tf.constant(global_scale, dtype=tf.float32),
        output_gain=output_gain,
        final_clip=final_clip
    ).numpy()

    mask = np.zeros_like(x_scaled, dtype=np.float32)
    mask[sensor_index, :, :] = 1.0

    observed = 1.0 - mask
    x_masked = x_scaled * observed

    x_model = np.concatenate([x_masked, observed], axis=-1)[None, ...]
    d_model = D_norm.astype(np.float32)[None, ...]

    preds = model.predict(
        {
            "seismic_input": x_model,
            "distance_input": d_model
        },
        verbose=0
    )

    y_pred_scaled = get_3branch_predictions(preds)[0]   # [S, T, 3]

    pred_mag = None
    if isinstance(preds, dict) and "MAG_Output" in preds:
        pred_mag = float(np.asarray(preds["MAG_Output"])[0, 0])
    elif isinstance(preds, list) and len(preds) >= 4:
        pred_mag = float(np.asarray(preds[3])[0, 0])

    if plot_inverse:
        true_event = inverse_global_asinh_scale(
            x_scaled,
            global_center,
            global_scale,
            output_gain
        )
        pred_event = inverse_global_asinh_scale(
            y_pred_scaled,
            global_center,
            global_scale,
            output_gain
        )
    else:
        true_event = x_scaled
        pred_event = y_pred_scaled

    true_comp = true_event[:, :, ci]   # [S, T]
    pred_comp = pred_event[:, :, ci]   # [S, T]

    reconstructed_comp = true_comp.copy()
    reconstructed_comp[sensor_index, :] = pred_comp[sensor_index, :]

    st_observed = Stream()
    st_reconstructed = Stream()

    for si, sta in enumerate(station_order):
        if sta not in site_dict:
            raise ValueError(f"Station {sta} not found in site file {site_file}")

        stla = site_dict[sta]["lat"]
        stlo = site_dict[sta]["lon"]
        stel = site_dict[sta]["elev_m"]

        tr_obs = Trace(data=true_comp[si].astype(np.float32))
        tr_obs.stats.station = sta
        tr_obs.stats.channel = chan
        tr_obs.stats.network = network
        tr_obs.stats.location = location
        tr_obs.stats.delta = 1.0 / sampling_rate
        tr_obs.stats.starttime = starttime
        tr_obs.stats.sac = {
            "stla": stla,
            "stlo": stlo,
            "stel": stel,
        }
        st_observed += tr_obs

        tr_rec = Trace(data=reconstructed_comp[si].astype(np.float32))
        tr_rec.stats.station = sta
        tr_rec.stats.channel = chan
        tr_rec.stats.network = network
        tr_rec.stats.location = location
        tr_rec.stats.delta = 1.0 / sampling_rate
        tr_rec.stats.starttime = starttime
        tr_rec.stats.sac = {
            "stla": stla,
            "stlo": stlo,
            "stel": stel,
        }
        st_reconstructed += tr_rec

    info = {
        "event_index": event_index,
        "sensor_index": sensor_index,
        "station": station_order[sensor_index],
        "component": chan,
        "starttime": starttime,
        "x_model": x_model,
        "d_model": d_model,
        "true_component_array": true_comp,
        "pred_component_array": pred_comp,
        "predicted_magnitude": pred_mag,
        "site_dict": site_dict,
    }

    return st_observed, st_reconstructed, info

def plot_array_geometry(st, figsize=(6,6), fname=None):

    x_km, y_km, lat0, lon0 = cardinal_fk.get_array_coordinates(st, return_centroid=True)

    print(f"Array centroid: lat={lat0:.4f}°, lon={lon0:.4f}°")
    print(f"Array aperture: Δx={x_km.max()-x_km.min():.3f} km, Δy={y_km.max()-y_km.min():.3f} km, Δ={np.sqrt((x_km.max()-x_km.min())**2 + (y_km.max()-y_km.min())**2):.3f} km")
    print(f"Number of sensors: {len(x_km)}")

    # Plot array geometry
    fig, ax = plt.subplots(figsize=figsize)

    ax.scatter(
        x_km, y_km,
        s=30,
        color='steelblue',
        edgecolor='black',
        linewidths=0.5,
        zorder=5
    )

    ax.scatter(
        0, 0,
        s=150,
        color='red',
        marker='+',
        zorder=6,
        linewidths=2.5,
        label='centroid'
    )

    # Add station labels
    for xi, yi, tr in zip(x_km, y_km, st):
        ax.text(
            xi + 0.01, yi + 0.01,
            tr.stats.station,
            fontsize=9,
            ha='left',
            va='bottom',
            color='black',
            zorder=7
        )

    ax.set_xlabel('x [km] (East)', fontsize=11)
    ax.set_ylabel('y [km] (North)', fontsize=11)
    ax.set_title(f'PFO Array Geometry ({len(x_km)} sensors)', fontsize=12, fontweight='bold')
    ax.set_aspect('equal')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if fname is not None:
        fig.savefig(fname, dpi=300)

    return x_km, y_km

def sliding_window_fk(st, x_km, y_km, freq_min=0.5, freq_max=5.0, smax=1.0, ngrid=200, window_length=10, overlap_percent=90):

    # Prepare data for sliding window (use full time range)
    fs = st[0].stats.sampling_rate
    data_all = np.array([tr.data.astype(float) for tr in st])

    print(f"Running sliding window F-K analysis...")
    print(f"  Window: {window_length}s, Overlap: {overlap_percent}%")
    print(f"  Note: First run may be slower due to JIT compilation")

    # Run sliding window F-K
    T_fk, B_fk, V_fk, S_fk = cardinal_fk.sliding_window_fk(
        data_all, x_km, y_km, fs,
        fmin=freq_min,
        fmax=freq_max,
        window_length=window_length,
        overlap_percent=overlap_percent,
        smax=smax,
        ngrid=ngrid,     
        detrend_data=False,
        apply_taper=False
    )

    print(f"\n✓ Complete!")

    return T_fk, B_fk, V_fk, S_fk

def plot_sliding_window_custom(st, element, T, B, V, C=None, v_min=0, v_max=5., 
                        semblance_threshold=None, twin_plot=None, clim=[0,1], figsize=(10,6),
                        baz_line=None, ylim_baz=None, fname=None):

    tr = st.select(station=element)[0]

    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(
        3, 2,
        width_ratios=[40, 1.0],
        height_ratios=[1, 1, 1],
        wspace=0.02,
        hspace=0.20
    )

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[1, 0], sharex=ax1)
    ax3 = fig.add_subplot(gs[2, 0], sharex=ax1)
    cax = fig.add_subplot(gs[1:, 1])  # colorbar only beside bottom two panels

    t_tr = np.arange(0, tr.stats.npts * tr.stats.delta, tr.stats.delta)
    ax1.plot(t_tr, tr.data / np.max(np.abs(tr.data)), 'k-')
    ax1.tick_params(labelbottom=False)
    ax1.set_ylabel('Normalized\nAmplitude')

    if C is not None:
        if semblance_threshold is not None:
            ix2 = np.where(C < semblance_threshold)
            ax2.scatter(T[ix2], B[ix2], s=0.05, c='lightgray')
            ix = np.where(C >= semblance_threshold)
            sc2 = ax2.scatter(T[ix], B[ix], s=4, c=C[ix], vmin=clim[0], vmax=clim[1], cmap='hot_r')
        else:
            sc2 = ax2.scatter(T, B, s=4, c=C, vmin=clim[0], vmax=clim[1], cmap='hot_r')
    else:
        ax2.plot(T, B, 'k.')
        sc2 = None

    if baz_line is not None:
        ax2.axhline(y=baz_line, color='blue', linestyle='--', linewidth=1, label=f'GC BAZ: {baz_line:.1f}°')
        ax2.legend(loc='upper right', fontsize=8)

    ax2.set_ylim(ylim_baz if ylim_baz is not None else [0, 360])
    ax2.set_ylabel('Backazimuth')
    if twin_plot is not None:
        ax2.set_xlim(twin_plot)
    ax2.tick_params(labelbottom=False)

    if C is not None:
        if semblance_threshold is not None:
            ix2 = np.where(C < semblance_threshold)
            ax3.scatter(T[ix2], V[ix2], s=0.05, c='lightgray')
            ix = np.where(C >= semblance_threshold)
            sc3 = ax3.scatter(T[ix], V[ix], s=4, c=C[ix], vmin=clim[0], vmax=clim[1], cmap='hot_r')
        else:
            sc3 = ax3.scatter(T, V, s=4, c=C, vmin=clim[0], vmax=clim[1], cmap='hot_r')
    else:
        ax3.plot(T, V, 'k.')

    ax3.set_ylim([v_min, v_max])
    ax3.set_ylabel('Phase vel.')
    ax3.set_xlabel('Time [s] after ' + str(tr.stats.starttime).split('.')[0].replace('T', ' '))
    ax3.set_xlim([t_tr[0], t_tr[-1]])

    if C is not None and sc2 is not None:
        fig.colorbar(sc2, cax=cax, label='Semblance')
        
    plt.tight_layout()
    if fname is not None:
        fig.savefig(fname, dpi=300)
    
    return fig, (ax1, ax2, ax3)

def plot_fk(st, x_km, y_km, time_start, time_end, freq_min=0.5, freq_max=5.0, smax=1.0, ngrid=400, fname=None):

    # Extract and prepare data
    time_start_fk = st[0].stats.starttime + time_start
    time_end_fk = st[0].stats.starttime + time_end
    st_fk = st.copy()
    st_fk.trim(starttime=time_start_fk, endtime=time_end_fk)

    # Get data array
    fs = st_fk[0].stats.sampling_rate
    min_length = min([tr.stats.npts for tr in st_fk])
    data_fk = np.array([tr.data[:min_length].astype(float) for tr in st_fk])

    print(f"Running F-K analysis on {time_end-time_start}s window...")

    # F-K analysis
    sx_vec, sy_vec, power_grid, semblance_grid = cardinal_fk.fk_analysis(
        data_fk, x_km, y_km, fs,
        freq_min, freq_max,
        smax=smax,
        ngrid=ngrid,
        detrend_data=False
    )


    # Plot results
    fig, axes, results = cardinal_fk.plot_fk(
        sx_vec, sy_vec, power_grid, semblance_grid,
        freq_min, freq_max,
        power_vmin_db=-8
    )
    
    if fname is not None:
        fig.savefig(fname, dpi=300)

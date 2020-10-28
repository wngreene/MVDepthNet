#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2020 Massachusetts Institute of Technology
#
# @file multi_view_test.py
# @author W. Nicholas Greene
# @date 2020-10-07 18:18:21 (Wed)

import os
import argparse

import yaml

import numpy as np

from PIL import Image

import matplotlib.pyplot as plt

import torch

import demon_dataset as dd

from depthNet_model import depthNet

import pytorch_utils
import depthmap_utils

BATCH_SIZE = 1

def get_depth_prediction_metrics(depthmap_true, depthmap_est):
    """Compute metrics commonly reported for KITTI depth prediction.

    Assumes no invalid inputs (i.e. mask has already been applied).

    Based on Monodepth.
    """
    thresh = np.maximum((depthmap_true / depthmap_est), (depthmap_est / depthmap_true))
    a1 = (thresh < 1.25   ).mean()
    a2 = (thresh < 1.25 ** 2).mean()
    a3 = (thresh < 1.25 ** 3).mean()

    rmse = (depthmap_true - depthmap_est) ** 2
    rmse = np.sqrt(rmse.mean())

    rmse_log = (np.log(depthmap_true) - np.log(depthmap_est)) ** 2
    rmse_log = np.sqrt(rmse_log.mean())

    abs_rel = np.mean(np.abs(depthmap_true - depthmap_est) / depthmap_true)

    sq_rel = np.mean(((depthmap_true - depthmap_est)**2) / depthmap_true)

    metrics = {"abs_rel": abs_rel,
               "sq_rel": sq_rel,
               "rmse": rmse,
               "rmse_log": rmse_log,
               "a1": a1,
               "a2": a2,
               "a3": a3}

    return metrics

def write_images(output_dir, image_idx, idepthmap_est, idepthmap_true):
    """Save colormapped depthmap images for debugging.
    """
    cmap = plt.get_cmap("magma")

    # idepthmaps.
    vmin = 0.0
    vmax = np.max(idepthmap_true)

    debug = np.squeeze(cmap((idepthmap_est - vmin) / (vmax - vmin)))
    debug_image = Image.fromarray(np.uint8(debug[:, :, :3] * 255))
    debug_image.save(os.path.join(output_dir, "idepthmap_{}_est.jpg".format(image_idx)))

    debug = np.squeeze(cmap((idepthmap_true - vmin) / (vmax - vmin)))
    debug_image = Image.fromarray(np.uint8(debug[:, :, :3] * 255))
    debug_image.save(os.path.join(output_dir, "idepthmap_{}_true.jpg".format(image_idx)))

    return

def write_losses_header(output_file, loss_dict):
    """Write header to losses file.
    """
    with open(output_file, "w") as ff:
        ff.write("file loss ")
        for key, value in loss_dict.items():
            if type(value) is list:
                for idx in range(len(value)):
                    ff.write("{}{} ".format(key, idx))
            else:
                ff.write("{} ".format(key))

        ff.write("\n")

    return

def write_losses(output_file, left_file, loss, loss_dict):
    """Write losses.
    """
    with open(output_file, "a") as ff:
        ff.write("{} {} ".format(left_file, loss))
        for key, value in loss_dict.items():
            if type(value) is list:
                for vv in value:
                    ff.write("{} ".format(vv.item()))
            else:
                ff.write("{} ".format(value.item()))

        ff.write("\n")

    return

def write_metrics_header(output_file, metrics_dict):
    """Write metrics header.
    """
    with open(output_file, "w") as ff:
        ff.write("file ")
        for key, value in metrics_dict.items():
            ff.write("{} ".format(key))
        ff.write("\n")
    return

def write_metrics(output_file, input_file, metrics_dict):
    """Write metrics as a line in output file.
    """
    with open(output_file, "a") as ff:
        ff.write("{} ".format(input_file))
        for key, value in metrics_dict.items():
            ff.write("{} ".format(value))
        ff.write("\n")

    return

def compute_avg_metrics(metrics_file):
    """Compute average metrics from metrics file.
    """
    keys = None
    with open(metrics_file, "r") as ff:
        header = ff.readline()
        keys = header.split()
    keys = keys[1:] # Skip filename.

    metrics = np.loadtxt(metrics_file, skiprows=1, usecols=range(1, len(keys) + 1))
    avg_metrics = np.mean(metrics, axis=0)

    avg_metrics_dict = {}
    for idx in range(len(keys)):
        avg_metrics_dict[keys[idx]] = avg_metrics[idx]

    avg_metrics_dict["num_samples"] = metrics.shape[0]

    return avg_metrics_dict

def get_groundtruth_depthmap(inputs):
    left_depthmap_true = inputs["left_depthmap_true"]
    left_depthmap_true = left_depthmap_true.squeeze().cpu().numpy()

    # Limits from DPSNet.
    min_depth = 0.5
    max_depth = 10.0

    return left_depthmap_true, min_depth, max_depth

def test(stereo_network, loader, save_images, output_dir):
    """Test network and compute metrics.
    """
    stereo_network.eval()

    batch_idx = 0
    # with torch.no_grad(): # No torch.no_grad in pytorch 0.3.
    for batch in loader:
        left_image = np.squeeze(batch["left_image"].cpu().numpy())
        right_image = np.squeeze(batch["right_image"][0].cpu().numpy())
        camera_k = batch["K"].cpu().numpy()[0, :3, :3]
        left2right = np.linalg.inv(batch["T_right_in_left"][0].cpu().numpy()[0, :, :])

        image_width = left_image.shape[-2]
        image_height = left_image.shape[-3]

        assert(image_width == 640)
        assert(image_height == 480)

        # for warp the image to construct the cost volume
        pixel_coordinate = np.indices([image_width, image_height]).astype(np.float32)
        pixel_coordinate = np.concatenate(
            (pixel_coordinate, np.ones([1, image_width, image_height])), axis=0)
        pixel_coordinate = np.reshape(pixel_coordinate, [3, -1])

        # convert to pythorch format
        torch_left_image = np.moveaxis(left_image, -1, 0)
        torch_left_image = np.expand_dims(torch_left_image, 0)
        torch_left_image = (torch_left_image - 81.0)/ 35.0
        torch_right_image = np.moveaxis(right_image, -1, 0)
        torch_right_image = np.expand_dims(torch_right_image, 0)
        torch_right_image = (torch_right_image - 81.0) / 35.0

        # process
        left_image_cuda = torch.Tensor(torch_left_image).cuda()
        left_image_cuda = torch.autograd.Variable(left_image_cuda, volatile=True)

        right_image_cuda = torch.Tensor(torch_right_image).cuda()
        right_image_cuda = torch.autograd.Variable(right_image_cuda, volatile=True)

        left_in_right_T = left2right[0:3, 3]
        left_in_right_R = left2right[0:3, 0:3]
        K = camera_k
        K_inverse = np.linalg.inv(K)
        KRK_i = K.dot(left_in_right_R.dot(K_inverse))
        KRKiUV = KRK_i.dot(pixel_coordinate)
        KT = K.dot(left_in_right_T)
        KT = np.expand_dims(KT, -1)
        KT = np.expand_dims(KT, 0)
        KT = KT.astype(np.float32)
        KRKiUV = KRKiUV.astype(np.float32)
        KRKiUV = np.expand_dims(KRKiUV, 0)
        KRKiUV_cuda_T = torch.Tensor(KRKiUV).cuda()
        KT_cuda_T = torch.Tensor(KT).cuda()

        tick, tock = pytorch_utils.start_timer()
        predict_depths = stereo_network(left_image_cuda, right_image_cuda, KRKiUV_cuda_T, KT_cuda_T)
        time_ms = pytorch_utils.stop_timer(tick, tock)

        print("runtime: {:.2f} ms (batch_size: {})".format(time_ms, BATCH_SIZE))

        # Convert idepthmap to depthmap.
        batch_left_idepthmap_est = predict_depths[0]
        for idx in range(batch_left_idepthmap_est.shape[0]):
            # Load groundtruth depthmaps.
            left_file = batch["left_filename"][idx]
            left_depthmap_true, min_depth, max_depth = get_groundtruth_depthmap(batch)
            left_idepthmap_true = np.copy(left_depthmap_true)
            left_idepthmap_true[left_idepthmap_true > 0] = 1.0 / left_idepthmap_true[left_idepthmap_true > 0]
            mask = (left_depthmap_true > min_depth) & (left_depthmap_true < max_depth)

            if np.sum(mask) <= 0:
                print("WARNING: No truth for image: {}".format(left_file))
                continue

            # Assume output is the same size as ground truth.
            left_idepthmap_est = batch_left_idepthmap_est[idx, :, :, :].unsqueeze(0)
            left_idepthmap_est = left_idepthmap_est.cpu().data.numpy().squeeze()

            left_depthmap_est = np.copy(left_idepthmap_est)
            left_depthmap_est[left_depthmap_est > 0] = 1.0 / left_depthmap_est[left_depthmap_est > 0]

            # Mask where truth and estimate are valid.
            mask = mask & (left_depthmap_est > min_depth) & (left_depthmap_est < max_depth)

            if save_images:
                left_dir, file_and_ext = os.path.split(left_file)
                left_dir = left_dir.replace(loader.dataset.data_dir, "") # Strip dataset prefix.
                left_output_dir = os.path.join(output_dir, left_dir[1:])
                image_num = os.path.splitext(file_and_ext)[0]
                if not os.path.exists(left_output_dir):
                    os.makedirs(left_output_dir)
                assert(os.path.exists(left_output_dir))
                write_images(left_output_dir, image_num,
                             left_idepthmap_est, left_idepthmap_true)
                left_dir_tokens = left_dir.split(os.path.sep)
                left_dir_tokens = [token for token in left_dir_tokens if token]

            # Compute depth metrics and write to file.
            depth_metrics_idx = get_depth_prediction_metrics(
                left_depthmap_true[mask], left_depthmap_est[mask])
            depth_metrics_file = os.path.join(output_dir, "depth_metrics.txt")
            if not os.path.exists(depth_metrics_file):
                write_metrics_header(depth_metrics_file, depth_metrics_idx)
            write_metrics(depth_metrics_file, left_file, depth_metrics_idx)

            # Save runtime metrics.
            runtime_metrics_file = os.path.join(output_dir, "runtime_metrics.txt")
            if not os.path.exists(runtime_metrics_file):
                with open(runtime_metrics_file, "w") as stream:
                    stream.write("file runtime_ms\n")
            with open(runtime_metrics_file, "a") as stream:
                stream.write("{} {}\n".format(left_file, time_ms))

            print("image: {}, ABS_REL: {:.2f}, A1: {:.2f}, A2: {:.2f}, A3: {:.2f}".format(
                left_file, depth_metrics_idx["abs_rel"], depth_metrics_idx["a1"],
                depth_metrics_idx["a2"], depth_metrics_idx["a3"]))

        print("Processed batch {}/{}".format(batch_idx, len(loader)))
        batch_idx += 1

    return

def load_data(test_split):
    """Load  dataset.
    """
    roll_right_image_180 = False
    testing_transforms = None
    if test_split == "demon1":
        data_dir = "/home/wng/Projects/data/demon/test"
        test_file = "demon_test.txt"
        dataset = dd.DeMoNDataset(data_dir, test_file,
                                  num_right_images=1, num_left_images=0,
                                  transform=testing_transforms)
    else:
        assert(False)

    loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE,
                                         shuffle=False, pin_memory=False)

    return loader

def main():
    """Tests MVDepthNet.
    """
    test_split = "demon1"

    # Load data.
    loader = load_data(test_split)

    # Load models.
    depthnet = depthNet([480, 640])
    model_data = torch.load('opensource_model.pth.tar')
    depthnet.load_state_dict(model_data['state_dict'])
    depthnet = depthnet.cuda()
    torch.backends.cudnn.benchmark = True
    depthnet.eval()

    # Create output dir.
    output_dir = os.path.join("depth_test_{}".format(test_split))
    assert(not os.path.exists(output_dir))
    os.makedirs(output_dir)

    # Evaluate network on test data.
    test(depthnet, loader, True, output_dir)

    # Compute metrics averaged across entire test set.
    avg_depth_metrics = compute_avg_metrics(os.path.join(output_dir, "depth_metrics.txt"))
    with open(os.path.join(output_dir, "avg_depth_metrics.txt"), "w") as ff:
        for key, value in avg_depth_metrics.items():
            ff.write("{}: {}\n".format(key, value))

    runtimes = np.loadtxt(os.path.join(output_dir, "runtime_metrics.txt"),
                          skiprows=1, usecols=1)
    mean_runtime = np.mean(runtimes)
    with open(os.path.join(output_dir, "avg_runtime_metrics.txt"), "w") as ff:
        ff.write("runtime_ms: {}\n".format(mean_runtime))
        ff.write("num_samples: {}\n".format(len(runtimes)))

    if "demon" in test_split:
        # Compute average metrics per scene type in demon.
        demon_types = ["mvs", "sun3d", "rgbd", "scenes11"]
        lines = []
        with open(os.path.join(output_dir, "depth_metrics.txt"), "r") as ff:
            lines = ff.readlines()

        header = lines[0]
        for demon_type in demon_types:
            metric_lines = [line for line in lines if demon_type in line]

            with open(os.path.join(output_dir, "depth_metrics_{}.txt".format(demon_type)), "w") as ff:
                ff.write(header)
                for line in metric_lines:
                    ff.write(line)

            avg_demon_metrics = compute_avg_metrics(os.path.join(output_dir, "depth_metrics_{}.txt".format(demon_type)))
            with open(os.path.join(output_dir, "avg_depth_metrics_{}.txt".format(demon_type)), "w") as ff:
                for key, value in avg_demon_metrics.items():
                    ff.write("{}: {}\n".format(key, value))

    return

if __name__ == '__main__':
    main()

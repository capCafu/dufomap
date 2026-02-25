"""
# Created: 2024-11-20 13:11
# Copyright (C) 2024-now, RPL, KTH Royal Institute of Technology
# Author: Qingwen Zhang  (https://kin-zhang.github.io/)
#
# This file is part of DUFOMap (https://github.com/KTH-RPL/dufomap) and
# DynamicMap Benchmark (https://github.com/KTH-RPL/DynamicMap_Benchmark) projects.
# If you find this repo helpful, please cite the respective publication as
# listed on the above website.

# Description: Output Cleaned Map through Python API.
"""

from pathlib import Path
import os, fire, time
from datetime import timedelta
import numpy as np
from tqdm import tqdm

from dufomap import dufomap
from dufomap.utils import pcdpy3

try:
    from scipy.spatial import cKDTree
except Exception:
    cKDTree = None


def inv_pose_matrix(pose):
    inv_pose = np.eye(4)
    inv_pose[:3, :3] = pose[:3, :3].T
    inv_pose[:3, 3] = -pose[:3, :3].T.dot(pose[:3, 3])
    return inv_pose


MIN_AXIS_RANGE = 0.2  # HARD CODED: remove ego vehicle points
MAX_AXIS_RANGE = 50  # HARD CODED: remove far away points
LAST_RUN_FRAMES = 0


class DynamicMapData:
    def __init__(self, directory):
        self.scene_id = directory.split("/")[-1]
        self.directory = Path(directory) / "pcd"
        self.pcd_files = [
            os.path.join(self.directory, f)
            for f in sorted(os.listdir(self.directory))
            if f.endswith(".pcd")
        ]

    def __len__(self):
        return len(self.pcd_files)

    def __getitem__(self, index_):
        res_dict = {
            "scene_id": self.scene_id,
            "timestamp": self.pcd_files[index_].split("/")[-1].split(".")[0],
        }
        pcd_ = pcdpy3.PointCloud.from_path(self.pcd_files[index_])
        pc0 = pcd_.np_data[:, :3]
        res_dict["pc"] = pc0.astype(np.float32)
        res_dict["pose"] = list(pcd_.viewpoint)
        return res_dict


def _range_mask(
    points: np.ndarray,
    pose: list,
    min_axis_range: float,
    max_axis_range: float,
) -> np.ndarray:
    norm_pc0 = np.linalg.norm(points[:, :3] - np.asarray(pose[:3]), axis=1)
    return (norm_pc0 > float(min_axis_range)) & (norm_pc0 < float(max_axis_range))


def _temporal_keep_mask(
    points: np.ndarray,
    prev_points: np.ndarray,
    next_points: np.ndarray,
    dist_thresh: float,
    mode: str,
) -> np.ndarray:
    if mode == "none" or dist_thresh <= 0.0 or points.size == 0:
        return np.ones(points.shape[0], dtype=bool)

    def _nn_mask(query_points: np.ndarray, ref_points: np.ndarray) -> np.ndarray:
        if ref_points.size == 0:
            return np.zeros(query_points.shape[0], dtype=bool)

        if cKDTree is not None:
            tree = cKDTree(ref_points, compact_nodes=False, balanced_tree=False)
            dists, _ = tree.query(query_points, k=1, workers=-1)
            return dists < float(dist_thresh)

    prev_ok = _nn_mask(points, prev_points)
    next_ok = _nn_mask(points, next_points)

    if mode == "both":
        return prev_ok & next_ok
    if mode == "either":
        return prev_ok | next_ok
    raise ValueError("temporal_mode must be one of: none, either, both")


def _ground_keep_mask(
    points: np.ndarray,
    ground_percentile: float,
    ground_margin: float,
) -> np.ndarray:
    if points.size == 0:
        return np.zeros(0, dtype=bool)
    z = points[:, 2]
    z_ref = np.percentile(z, float(ground_percentile))
    return z <= (z_ref + float(ground_margin))


def _param_token(value) -> str:
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, float):
        value = np.format_float_positional(float(value), trim="-")
    return str(value).replace("-", "m").replace(".", "p")


def _run_output_stem(
    data_dir: str,
    output_name: str,
    resolution: float,
    inflate_hits_dist: float,
    inflate_unknown: int,
    min_axis_range: float,
    max_axis_range: float,
    temporal_mode: str,
    temporal_dist_thresh: float,
    temporal_step: int,
    preserve_ground: bool,
    ground_percentile: float,
    ground_margin: float,
    voxel_map: bool,
    num_threads: int,
) -> str:
    data_path = Path(data_dir).expanduser().resolve()
    scene_id = data_path.name or "scene"
    prefix = (Path(output_name).stem or "dufomap_output").replace(" ", "_")
    mode_token = "".join(ch if ch.isalnum() else "_" for ch in str(temporal_mode))

    parts = [
        prefix,
        scene_id,
        f"res{_param_token(resolution)}",
        f"ds{_param_token(inflate_hits_dist)}",
        f"dp{_param_token(inflate_unknown)}",
        f"min{_param_token(min_axis_range)}",
        f"max{_param_token(max_axis_range)}",
        f"tm{mode_token}",
        f"td{_param_token(temporal_dist_thresh)}",
        f"ts{_param_token(temporal_step)}",
        f"pg{_param_token(preserve_ground)}",
        f"gp{_param_token(ground_percentile)}",
        f"gm{_param_token(ground_margin)}",
        f"vox{_param_token(voxel_map)}",
        f"thr{_param_token(num_threads)}",
    ]

    return str(data_path / "_".join(parts))


def main_vis(
    data_dir: str = "/home/kin/data/00",
    voxel_map: bool = False,  # output voxel-level map or raw point-level.
    resolution: float = 0.1,  # voxel size (meters)
    inflate_hits_dist: float = 0.1,  # d_s
    inflate_unknown: int = 1,  # d_p
    min_axis_range: float = 1.0,  # remove points close to sensor
    max_axis_range: float = 100.0,  # remove far points
    temporal_dist_thresh: float = 0.3,  # temporal NN threshold (meters)
    temporal_step: int = 5,  # compare j against j-step and j+step
    temporal_mode: str = "both",  # one of: none, either, both
    preserve_ground: bool = True,  # keep low-Z points through temporal filtering
    ground_percentile: float = 16.0,  # low-Z percentile for ground proxy
    ground_margin: float = 0.15,  # meters above the low-Z percentile
    output_name: str = "dufomap_output",  # output basename used by dufomap API
    num_threads: int = 16,
):
    dataset = DynamicMapData(data_dir)
    global LAST_RUN_FRAMES
    LAST_RUN_FRAMES = len(dataset)

    # STEP 0: initialize
    mydufo = dufomap(
        float(resolution),
        float(inflate_hits_dist),
        int(inflate_unknown),
        num_threads=int(num_threads),
    )
    cloud_acc_chunks = []
    frame_cache = {}

    if temporal_mode != "none" and cKDTree is None:
        print(
            "[WARN] scipy not found. Temporal consistency uses coarse voxel fallback.",
            flush=True,
        )

    def _get_data(idx: int):
        if idx not in frame_cache:
            frame_cache[idx] = dataset[idx]
        return frame_cache[idx]

    for data_id in (pbar := tqdm(range(0, len(dataset)), ncols=100)):
        data = _get_data(data_id)
        now_scene_id = data["scene_id"]
        pbar.set_description(
            f"id: {data_id}, scene_id: {now_scene_id}, timestamp: {data['timestamp']}"
        )

        range_mask = _range_mask(
            data["pc"],
            data["pose"],
            min_axis_range=min_axis_range,
            max_axis_range=max_axis_range,
        )
        points_to_integrate = data["pc"][range_mask]

        use_temporal = (
            temporal_mode != "none"
            and temporal_step > 0
            and (data_id - temporal_step) >= 0
            and (data_id + temporal_step) < len(dataset)
        )
        if use_temporal and points_to_integrate.size:
            prev_data = _get_data(data_id - temporal_step)
            next_data = _get_data(data_id + temporal_step)

            prev_points = prev_data["pc"][
                _range_mask(
                    prev_data["pc"],
                    prev_data["pose"],
                    min_axis_range=min_axis_range,
                    max_axis_range=max_axis_range,
                )
            ]
            next_points = next_data["pc"][
                _range_mask(
                    next_data["pc"],
                    next_data["pose"],
                    min_axis_range=min_axis_range,
                    max_axis_range=max_axis_range,
                )
            ]
            keep_mask = _temporal_keep_mask(
                points_to_integrate,
                prev_points,
                next_points,
                dist_thresh=temporal_dist_thresh,
                mode=temporal_mode,
            )
            if preserve_ground:
                keep_mask |= _ground_keep_mask(
                    points_to_integrate,
                    ground_percentile=ground_percentile,
                    ground_margin=ground_margin,
                )
            points_to_integrate = points_to_integrate[keep_mask]

        # STEP 1: integrate point cloud into dufomap
        if points_to_integrate.size:
            mydufo.run(points_to_integrate, data["pose"], cloud_transform=False)
        # STEP 1: collect points for final output map.
        if points_to_integrate.size:
            cloud_acc_chunks.append(points_to_integrate)

        # Keep cache bounded.
        drop_idx = data_id - temporal_step - 1
        frame_cache.pop(drop_idx, None)

    # STEP 2: propagate
    mydufo.oncePropagateCluster(if_propagate=True, if_cluster=False)
    # STEP 3: Map results; You can save the voxel map directly based on the resolution we set before:
    if cloud_acc_chunks:
        cloud_acc = np.concatenate(cloud_acc_chunks, axis=0).astype(
            np.float32, copy=False
        )
    else:
        cloud_acc = np.zeros((0, 3), dtype=np.float32)

    output_stem = _run_output_stem(
        data_dir=data_dir,
        output_name=output_name,
        resolution=resolution,
        inflate_hits_dist=inflate_hits_dist,
        inflate_unknown=inflate_unknown,
        min_axis_range=min_axis_range,
        max_axis_range=max_axis_range,
        temporal_mode=temporal_mode,
        temporal_dist_thresh=temporal_dist_thresh,
        temporal_step=temporal_step,
        preserve_ground=preserve_ground,
        ground_percentile=ground_percentile,
        ground_margin=ground_margin,
        voxel_map=voxel_map,
        num_threads=num_threads,
    )
    mydufo.outputMap(cloud_acc, voxel_map=voxel_map, file_name=output_stem)

    mydufo.printDetailTiming()


if __name__ == "__main__":
    start_time = time.time()
    fire.Fire(main_vis)
    elapsed = time.time() - start_time
    print(f"Time used: {elapsed:.2f} s")
    if LAST_RUN_FRAMES > 0 and elapsed > 0:
        print(
            f"Speed: {LAST_RUN_FRAMES / elapsed:.2f} Hz, {elapsed / LAST_RUN_FRAMES:.4f} s/frame"
        )
    else:
        print("Speed: n/a Hz, n/a s/frame")
    print(f"Time used (H:MM:SS): {timedelta(seconds=int(elapsed))}")

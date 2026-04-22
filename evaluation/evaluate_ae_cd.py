import os
import glob
import h5py
import numpy as np
import argparse
from joblib import Parallel, delayed
import random
from scipy.spatial import cKDTree as KDTree
import time
import sys
sys.path.append("..")
from utils import read_ply
from cadlib.visualize import vec2CADsolid, CADsolid2pc


PC_ROOT = "../data/pc_cad"
# data that is unable to process
SKIP_DATA = [""]

'''每个生成点 p 在真实点云中找最近邻 q，计算距离平方；
每个真实点 q 在生成点云中找最近邻 p；
两部分求平均后相加。
Chamfer Distance 越小，说明模型生成的 3D 形状越接近真实形状'''
def chamfer_dist(gt_points, gen_points, offset=0, scale=1):
    gen_points = gen_points / scale - offset

    # one direction
    gen_points_kd_tree = KDTree(gen_points)
    one_distances, one_vertex_ids = gen_points_kd_tree.query(gt_points)
    gt_to_gen_chamfer = np.mean(np.square(one_distances))

    # other direction
    gt_points_kd_tree = KDTree(gt_points)
    two_distances, two_vertex_ids = gt_points_kd_tree.query(gen_points)
    gen_to_gt_chamfer = np.mean(np.square(two_distances))

    return gt_to_gen_chamfer + gen_to_gt_chamfer

'''
点云归一化
当生成的点云坐标超出 [-1, 1] 范围时，将其缩放到单位立方体内，防止偏移或过大'''
def normalize_pc(points):
    scale = np.max(np.abs(points))
    points = points / scale
    return points


def process_one(path):
    with h5py.File(path, 'r') as fp:
        out_vec = fp["out_vec"][:].astype(np.float64)
        # gt_vec = fp["gt_vec"][:].astype(np.float)

    #根据文件名找到对应的 Ground Truth 点云
    data_id = path.split('/')[-1].split('.')[0][:8]
    truck_id = data_id[:4]
    gt_pc_path = os.path.join(PC_ROOT, truck_id, data_id + '.ply')
    if not os.path.exists(gt_pc_path):
        return None

    #从 out_vec 重建 CAD 几何体
    try:
        shape = vec2CADsolid(out_vec)
    except Exception as e:
        print("create_CAD failed", data_id)
        return None

    #CAD 实体 → 点云采样
    try:
        out_pc = CADsolid2pc(shape, args.n_points, data_id)
    except Exception as e:
        print("convert pc failed:", data_id)
        return None

    if np.max(np.abs(out_pc)) > 2: # normalize out-of-bound data
        out_pc = normalize_pc(out_pc)

    #读取 Ground Truth 点云
    gt_pc = read_ply(gt_pc_path)
    sample_idx = random.sample(list(range(gt_pc.shape[0])), args.n_points)
    gt_pc = gt_pc[sample_idx]

    #计算 Chamfer Distance
    cd = chamfer_dist(gt_pc, out_pc)
    return cd


def run(args):
    filepaths = sorted(glob.glob(os.path.join(args.src, "*.h5")))
    if args.num != -1:
        filepaths = filepaths[:args.num]

    save_path = args.src + '_pc_stat.txt'
    record_res = None
    if os.path.exists(save_path):
        response = input(save_path + ' already exists, overwrite? (y/n) ')
        if response == 'y':
            os.system("rm {}".format(save_path))
            record_res = None
        else:
            with open(save_path, 'r') as fp:
                record_res = fp.readlines()
                n_processed = len(record_res) - 3

    if args.parallel:
        dists = Parallel(n_jobs=8, verbose=2)(delayed(process_one)(x) for x in filepaths)
    else:
        dists = []
        for i in range(len(filepaths)):
            print("processing[{}] {}".format(i, filepaths[i]))
            data_id = filepaths[i].split('/')[-1].split('.')[0]

            if record_res is not None and i < n_processed:
                record_dist = record_res[i].split('\t')[-1][:-1]
                record_dist = None if record_dist == 'None' else eval(record_dist)
                dists.append(record_dist)
                continue

            if data_id in SKIP_DATA:
                print("skip {}".format(data_id))
                res = None
            else:
                res = process_one(filepaths[i])
            with open(save_path, 'a') as fp:
                print("{}\t{}\t{}".format(i, data_id, res), file=fp)
            dists.append(res)

    valid_dists = [x for x in dists if x is not None]
    valid_dists = sorted(valid_dists)
    print("top 20 largest error:")
    print(valid_dists[-20:][::-1])
    n_valid = len(valid_dists)
    n_invalid = len(dists) - n_valid

    avg_dist = np.mean(valid_dists)
    trim_avg_dist = np.mean(valid_dists[int(n_valid * 0.1):-int(n_valid * 0.1)])
    med_dist = np.median(valid_dists)

    print("#####" * 10)
    print("total:", len(filepaths), "\t invalid:", n_invalid, "\t invalid ratio:", n_invalid / len(filepaths))
    print("avg dist:", avg_dist, "trim_avg_dist:", trim_avg_dist, "med dist:", med_dist)
    with open(save_path, "a") as fp:
        print("#####" * 10, file=fp)
        print("total:", len(filepaths), "\t invalid:", n_invalid, "\t invalid ratio:", n_invalid / len(filepaths),
              file=fp)
        print("avg dist:", avg_dist, "trim_avg_dist:", trim_avg_dist, "med dist:", med_dist,
              file=fp)


parser = argparse.ArgumentParser()
parser.add_argument('--src', type=str, default=None, required=True)
parser.add_argument('--n_points', type=int, default=2000)
parser.add_argument('--num', type=int, default=-1)
parser.add_argument('--parallel', action='store_true', help="use parallelization")
args = parser.parse_args()

print(args.src)
print("SKIP DATA:", SKIP_DATA)
since = time.time()
run(args)
end = time.time()
print("running time: {}s".format(end - since))

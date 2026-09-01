# -*- coding: utf-8 -*-
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


import time
import os
import gc
import json
import argparse
from pathlib import Path
import numpy as np
import scipy.io
from scipy.spatial import Delaunay
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rcParams["figure.dpi"] = 200
from itertools import chain
import torch
import torch.nn as nn
import torch.optim as optim
from utils.kan_efficiency import *

from utils.gridPlot2D import (scatterPlot, genGrid, plotDispStrainEnerg_uni,
                              plotPhiStrainEnerg_uni, plotConvergence,
                              createFolder, plot1dPhi, plotForceDisp)

# -----------------------------------------------------------------------------
# Helper for deterministic runs (equivalent to np.random.seed / tf.set_random_seed)
# -----------------------------------------------------------------------------
seed = 2025
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available() and os.environ.get("CUDA_VISIBLE_DEVICES", None) not in ("", "-1"):
    torch.cuda.manual_seed_all(seed)

# Default is CPU-safe. If CUDA is visible and available, GPU can be used.
device = torch.device("cuda" if (torch.cuda.is_available() and os.environ.get("CUDA_VISIBLE_DEVICES", None) not in ("", "-1")) else "cpu")

# Respect SLURM CPU allocation when running on CPU.
try:
    n_threads = int(os.environ.get("SLURM_CPUS_PER_TASK", os.environ.get("OMP_NUM_THREADS", "0")))
    if n_threads > 0:
        torch.set_num_threads(n_threads)
except Exception:
    pass

print("Using device:", device)
print("PyTorch CPU threads:", torch.get_num_threads())
# -----------------------------------------------------------------------------
#   Neural‑network core (unchanged public API, now inheriting nn.Module)
# -----------------------------------------------------------------------------

class MultiLayerNet(torch.nn.Module):
    def __init__(self, D_in, H, D_out):
        """
        In the constructor we instantiate two nn.Linear modules and assign them as
        member variables.
        """
        super(MultiLayerNet, self).__init__()
        self.linear1 = torch.nn.Linear(D_in, H)
        self.linear2 = torch.nn.Linear(H, H)
        self.linear3 = torch.nn.Linear(H, H)
        self.linear4 = torch.nn.Linear(H, H)
        self.linear5 = torch.nn.Linear(H, D_out)

        torch.nn.init.normal_(self.linear1.bias, mean=0, std=1)
        torch.nn.init.normal_(self.linear2.bias, mean=0, std=1)
        torch.nn.init.normal_(self.linear3.bias, mean=0, std=1)
        torch.nn.init.normal_(self.linear4.bias, mean=0, std=1)
        torch.nn.init.normal_(self.linear5.bias, mean=0, std=1)

        torch.nn.init.normal_(self.linear1.weight, mean=0, std=np.sqrt(2/(D_in+H)))
        torch.nn.init.normal_(self.linear2.weight, mean=0, std=np.sqrt(2/(H+H)))
        torch.nn.init.normal_(self.linear3.weight, mean=0, std=np.sqrt(2/(H+H)))
        torch.nn.init.normal_(self.linear4.weight, mean=0, std=np.sqrt(2/(H+H)))
        torch.nn.init.normal_(self.linear5.weight, mean=0, std=np.sqrt(2/(H+D_out)))

    def forward(self, x):
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Tensors.
        """
        yt = x
        y1 = torch.tanh(self.linear1(yt))
        y2 = torch.tanh(self.linear2(y1))
        y3 = torch.tanh(self.linear3(y2)) + y1
        y4 = torch.tanh(self.linear4(y3)) + y2
        y =  self.linear5(y4)
        return y
    

# -----------------------------------------------------------------------------
#  Problem‑specific subclass with boundary conditions (name unchanged)
# -----------------------------------------------------------------------------
class DEM_PF(nn.Module):
    """Tension-plate PINN for horizontal left-edge crack under top y-loading (PyTorch)."""

    def __init__(self, model, modelNN_U, modelNN_phi):
        super(DEM_PF, self).__init__()
        self.model = model
        self.modelNN_U = modelNN_U
        self.modelNN_phi = modelNN_phi
        self.crackTip = 0.5  # moved here so it's available immediately

        self.E = model["E"]
        self.nu = model["nu"]
        # 这个材料系数说明是平面应变问题
        self.c11 = self.E * (1 - self.nu) / ((1 + self.nu) * (1 - 2 * self.nu))
        self.c22 = self.c11
        self.c12 = self.E * self.nu / ((1 + self.nu) * (1 - 2 * self.nu))
        self.c21 = self.c12
        self.c31 = 0.0
        self.c32 = 0.0
        self.c13 = 0.0
        self.c23 = 0.0
        self.c33 = self.E / (2 * (1 + self.nu))

        self.lamda = self.E * self.nu / ((1 - 2 * self.nu) * (1 + self.nu))
        self.mu = 0.5 * self.E / (1 + self.nu)

        # Fracture energy from JSON/XDEM material model.
        self.cEnerg = model.get("Gc", 2.7)
        self.B = model.get("B", 1000.0)
        self.l = model["l"]

        # Crack geometry.
        # Default geometry is the attached image:
        # horizontal left-edge crack from (0, crack_y) to (crack_fraction, crack_y).
        self.crack_type = model.get("crack_type", "edge_horizontal")
        self.crack_fraction = model.get("crack_fraction", 0.50)
        self.crack_y = model.get("crack_y", 0.50)
        self.crack_x_start = model.get("crack_x_start", 0.0)
        self.crack_tip_radius_norm = model.get("crack_tip_radius_norm", 0.0)

        # These are kept only for backward compatibility if you still want to
        # test diagonal/cross cracks with this same file.
        self.crack_angle_deg = model.get("crack_angle_deg", 0.0)
        self.second_crack_angle_deg = model.get("second_crack_angle_deg", -self.crack_angle_deg)
        self.crack_center = model.get("crack_center", [0.5, 0.5])
        self.crack_width_norm = model.get("crack_width_norm", self.l)

        self.lb = torch.tensor(model["lb"], dtype=torch.float32, device=device)
        self.ub = torch.tensor(model["ub"], dtype=torch.float32, device=device)


    # net_uv is already implemented identically in base class; override here to
    # keep attribute order if you had custom logic.  Keep original code 1‑to‑1.
    def net_uv(self, x, y, vdelta):
        X = torch.cat([x, y], dim=1)
        H = 2.0 * (X - self.lb) / (self.ub - self.lb) - 1.0 # 归一化到-1到1
        uv = self.modelNN_U(H)
        

        

        
        # Combine analytical solution with neural network prediction
        uNN = uv[:, 0:1]
        vNN = uv[:, 1:2]
        
        # Add analytical solution to neural network prediction
        u = y * (uNN)
        v = y * (y - 1) * (vNN) + y * vdelta
        
        return u, v

    def net_phi(self, x, y):
        X = torch.cat([x, y], dim=1)
        H = 2.0 * (X - self.lb) / (self.ub - self.lb) - 1.0 # 归一化到-1到1
        phi = self.modelNN_phi(H)

        return phi

    # net_hist for horizontal left-edge crack
    def net_hist(self, x, y):
        """
        Initial history field in normalized coordinates.

        Default:
            crack_type = "edge_horizontal"

        Geometry corresponding to the attached image:
            domain:       0 <= x <= 1, 0 <= y <= 1
            crack start:  (crack_x_start, crack_y)
            crack tip:    (crack_x_start + crack_fraction, crack_y)

        With crack_fraction = 0.50 and crack_y = 0.50, this gives a horizontal
        left-edge crack of normalized length 0.5 at the mid-height of the plate.

        Backward-compatible options:
            crack_type = "diagonal"
            crack_type = "cross"
        """
        shape = x.shape
        init_hist = torch.zeros(shape, dtype=torch.float32, device=device)

        crack_width = float(self.crack_width_norm)

        def apply_crack_from_distance(dist):
            mask = dist < 0.5 * crack_width
            if torch.any(mask):
                shape_fun = torch.clamp(1.0 - 2.0 * dist[mask] / crack_width, min=0.0)
                init_hist[mask] = self.B * self.cEnerg * 0.5 * shape_fun / self.l

        def distance_to_segment_xy(p1x, p1y, p2x, p2y):
            vx = p2x - p1x
            vy = p2y - p1y
            vv = vx * vx + vy * vy + 1.0e-20

            t = ((x - p1x) * vx + (y - p1y) * vy) / vv
            t = torch.clamp(t, 0.0, 1.0)

            closest_x = p1x + t * vx
            closest_y = p1y + t * vy

            dist = torch.sqrt((x - closest_x) ** 2 + (y - closest_y) ** 2 + 1.0e-20)
            return dist

        if self.crack_type in ["edge_horizontal", "horizontal_edge", "edge"]:
            # Horizontal left-edge crack: from x=0 to x=a at y=0.5 by default.
            p1x = torch.tensor(float(self.crack_x_start), dtype=torch.float32, device=device)
            p1y = torch.tensor(float(self.crack_y), dtype=torch.float32, device=device)
            p2x = torch.tensor(float(self.crack_x_start + self.crack_fraction), dtype=torch.float32, device=device)
            p2y = torch.tensor(float(self.crack_y), dtype=torch.float32, device=device)

            # Keep the crack inside the normalized domain.
            p2x = torch.clamp(p2x, 0.0, 1.0)

            dist = distance_to_segment_xy(p1x, p1y, p2x, p2y)

            # Optional stronger history at the crack tip.
            # Usually keep crack_tip_radius_norm = 0.0 for phase-field/XDEM.
            if float(self.crack_tip_radius_norm) > 0.0:
                dist_tip = torch.sqrt((x - p2x) ** 2 + (y - p2y) ** 2 + 1.0e-20)
                dist = torch.minimum(dist, dist_tip)

            apply_crack_from_distance(dist)
            return init_hist

        # ------------------------------------------------------------
        # Backward compatibility: central diagonal/cross crack
        # ------------------------------------------------------------
        cx = torch.tensor(float(self.crack_center[0]), dtype=torch.float32, device=device)
        cy = torch.tensor(float(self.crack_center[1]), dtype=torch.float32, device=device)
        half_len = 0.5 * float(self.crack_fraction)

        def distance_to_segment_angle(angle_deg):
            beta = torch.tensor(np.deg2rad(angle_deg), dtype=torch.float32, device=device)
            dx = torch.cos(beta)
            dy = torch.sin(beta)

            p1x = cx - half_len * dx
            p1y = cy - half_len * dy
            p2x = cx + half_len * dx
            p2y = cy + half_len * dy

            return distance_to_segment_xy(p1x, p1y, p2x, p2y)

        dist1 = distance_to_segment_angle(self.crack_angle_deg)

        if self.crack_type == "cross":
            dist2 = distance_to_segment_angle(self.second_crack_angle_deg)
            dist = torch.minimum(dist1, dist2)
        else:
            dist = dist1

        apply_crack_from_distance(dist)
        return init_hist


    # --- update history tensor -----------------------------------------------------
    #二维有拉压能量分解啊
    def net_update_hist(self, x, y, u_x, v_y, u_xy, hist):
        init_hist = self.net_hist(x, y)
        # tensile strain energy density
        u_xy = 0.5 * u_xy
        eigSum = u_x + v_y # 应变的迹
        sEnergy_pos = 0.125 * self.lamda * (eigSum + torch.abs(eigSum)) ** 2 + \
                      0.25 * self.mu * ((u_x + torch.abs(u_x)) ** 2 + (v_y + torch.abs(v_y)) ** 2 + 2*(u_xy+ torch.abs(u_xy)) ** 2)
        hist_temp = torch.maximum(init_hist, sEnergy_pos)
        hist = torch.maximum(hist, hist_temp)
        return hist

    def net_energy(self, x, y, hist, vdelta):
        # enable gradients
        x.requires_grad_(True)
        y.requires_grad_(True)

        u, v = self.net_uv(x, y, vdelta)
        phi = self.net_phi(x, y)
        g = (1 - phi) ** 2
        phi_x = torch.autograd.grad(phi, x, grad_outputs=torch.ones_like(phi), create_graph=True)[0]
        phi_y = torch.autograd.grad(phi, y, grad_outputs=torch.ones_like(phi), create_graph=True)[0]
        nabla = phi_x ** 2 + phi_y ** 2

        u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        v_y = torch.autograd.grad(v, y, grad_outputs=torch.ones_like(v), create_graph=True)[0]
        u_y = torch.autograd.grad(u, y, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        v_x = torch.autograd.grad(v, x, grad_outputs=torch.ones_like(v), create_graph=True)[0]
        u_xy = (u_y + v_x)
        
        # if iStep == 1: # 只在第一步迭代步下调用初始场
        #     hist = self.net_hist(x, y)
        # hist = self.net_update_hist(x, y, u_x, v_y, u_xy, hist) # 不让历史场更新

        # sigmaX = self.c11 * u_x + self.c12 * v_y
        # sigmaY = self.c21 * u_x + self.c22 * v_y
        # tauXY = self.c33 * u_xy

        u_xy = 0.5 * u_xy
        eigSum = u_x + v_y # 应变的迹
        sEnergy_pos = 0.125 * self.lamda * (eigSum + torch.abs(eigSum)) ** 2 + \
                      0.25 * self.mu * ((u_x + torch.abs(u_x)) ** 2 + (v_y + torch.abs(v_y)) ** 2 + 2*(u_xy + torch.abs(u_xy)) ** 2)
        sEnergy_neg = 0.125 * self.lamda * (eigSum - torch.abs(eigSum)) ** 2 + \
                      0.25 * self.mu * ((u_x - torch.abs(u_x)) ** 2 + (v_y - torch.abs(v_y)) ** 2 + 2*(u_xy - torch.abs(u_xy)) ** 2)

        energy_u = g * sEnergy_pos + sEnergy_neg
        energy_phi = 0.5 * self.cEnerg * (phi ** 2 / self.l + self.l * nabla) + g * hist
        return energy_u, energy_phi, hist.detach()  # detach hist to stop gradient here


    def net_energy_update_his(self, x, y, hist, vdelta):
        # enable gradients
        x.requires_grad_(True)
        y.requires_grad_(True)

        u, v = self.net_uv(x, y, vdelta)
        phi = self.net_phi(x, y)
        g = (1 - phi) ** 2
        phi_x = torch.autograd.grad(phi, x, grad_outputs=torch.ones_like(phi), create_graph=True)[0]
        phi_y = torch.autograd.grad(phi, y, grad_outputs=torch.ones_like(phi), create_graph=True)[0]
        nabla = phi_x ** 2 + phi_y ** 2

        u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        v_y = torch.autograd.grad(v, y, grad_outputs=torch.ones_like(v), create_graph=True)[0]
        u_y = torch.autograd.grad(u, y, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        v_x = torch.autograd.grad(v, x, grad_outputs=torch.ones_like(v), create_graph=True)[0]
        u_xy = (u_y + v_x)
        
        hist = self.net_update_hist(x, y, u_x, v_y, u_xy, hist) 

        # sigmaX = self.c11 * u_x + self.c12 * v_y
        # sigmaY = self.c21 * u_x + self.c22 * v_y
        # tauXY = self.c33 * u_xy

        u_xy = 0.5 * u_xy
        eigSum = u_x + v_y # 应变的迹
        sEnergy_pos = 0.125 * self.lamda * (eigSum + torch.abs(eigSum)) ** 2 + \
                      0.25 * self.mu * ((u_x + torch.abs(u_x)) ** 2 + (v_y + torch.abs(v_y)) ** 2 + 2*(u_xy + torch.abs(u_xy)) ** 2)
        sEnergy_neg = 0.125 * self.lamda * (eigSum - torch.abs(eigSum)) ** 2 + \
                      0.25 * self.mu * ((u_x - torch.abs(u_x)) ** 2 + (v_y - torch.abs(v_y)) ** 2 + 2*(u_xy - torch.abs(u_xy)) ** 2)

        energy_u = g * sEnergy_pos + sEnergy_neg
        energy_phi = 0.5 * self.cEnerg * (phi ** 2 / self.l + self.l * nabla) + g * hist
        return energy_u, energy_phi, hist.detach()  # detach hist to stop gradient here

    def net_traction(self, x, y, vdelta):
        x.requires_grad_(True)
        y.requires_grad_(True)
        u, v = self.net_uv(x, y, vdelta)
        u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        v_y = torch.autograd.grad(v, y, grad_outputs=torch.ones_like(v), create_graph=True)[0]
        traction = self.c21 * u_x + self.c22 * v_y
        return traction

    def train_model(self, X_f, v_delta, hist_f, nIter, his_deal):
        
        # keep identical attribute names used in plotting utils
        self.loss_adam_buff = np.zeros(nIter)
        self.lbfgs_buffer = []
        # convert data to torch tensors on device
        x_f = torch.tensor(X_f[:, 0:1], dtype=torch.float32, device=device)
        y_f = torch.tensor(X_f[:, 1:2], dtype=torch.float32, device=device)
        wt_f = torch.tensor(X_f[:, 2:3], dtype=torch.float32, device=device)
        hist = torch.tensor(hist_f, dtype=torch.float32, device=device)
        vdelta = torch.tensor(v_delta, dtype=torch.float32, device=device)

        def loss_fn(his_deal):
            if his_deal == 'fix':
                energy_u_pred, energy_phi_pred, _ = self.net_energy(x_f, y_f, hist, vdelta)
            if his_deal == 'update':
                energy_u_pred, energy_phi_pred, _ = self.net_energy_update_his(x_f, y_f, hist, vdelta)
            loss_energy_u = torch.sum(energy_u_pred * wt_f)
            loss_energy_phi = torch.sum(energy_phi_pred * wt_f)
            return loss_energy_u + loss_energy_phi, loss_energy_u, loss_energy_phi

        # ----------------- Adam phase -----------------------------   
        optimizer_U_adam = optim.Adam(self.modelNN_U.parameters(), lr=1e-3)
        optimizer_phi_adam = optim.Adam(self.modelNN_phi.parameters(), lr=1e-3)
        t0 = time.time()
        for it in range(nIter):
            optimizer_U_adam.zero_grad()
            optimizer_phi_adam.zero_grad()  
            loss, e_u, e_phi = loss_fn(his_deal)
            loss.backward()
            optimizer_U_adam.step()
            optimizer_phi_adam.step()
            self.loss_adam_buff[it] = loss.item()
            if it % 100 == 0:
                dt = time.time() - t0
                print(f"It: {it}, Total Loss: {loss.item():.3e}, Energy U: {e_u.item():.3e}, "
                      f"Energy Phi: {e_phi.item():.3e}, Time: {dt:.2f}")
                t0 = time.time()



    def predict(self, X_star, Hist_star, v_delta):
        x = torch.tensor(X_star[:, 0:1], dtype=torch.float32, device=device)
        y = torch.tensor(X_star[:, 1:2], dtype=torch.float32, device=device)
        
        
        hist = torch.tensor(Hist_star, dtype=torch.float32, device=device)
        vdelta = torch.tensor(v_delta, dtype=torch.float32, device=device)

        u_pred, v_pred = self.net_uv(x, y, vdelta)
        phi_pred = self.net_phi(x, y)
        energy_u_pred, energy_phi_pred, hist_pred = self.net_energy_update_his(x, y, hist, vdelta)

        return (u_pred.detach().cpu().numpy(), v_pred.detach().cpu().numpy(),
                         phi_pred.detach().cpu().numpy(),
                         energy_u_pred.detach().cpu().numpy(),
                         energy_phi_pred.detach().cpu().numpy(),
                         hist_pred.detach().cpu().numpy())


    def predict_traction(self, X_star, v_delta):
        x = torch.tensor(X_star[:, 0:1], dtype=torch.float32, device=device)
        y = torch.tensor(X_star[:, 1:2], dtype=torch.float32, device=device)
        vdelta = torch.tensor(v_delta, dtype=torch.float32, device=device)
        trac = self.net_traction(x, y, vdelta)
        return trac.detach().cpu().numpy()


    def predict_phi(self, X_star):
        x = torch.tensor(X_star[:, 0:1], dtype=torch.float32, device=device)
        y = torch.tensor(X_star[:, 1:2], dtype=torch.float32, device=device)
        phi = self.net_phi(x, y)
        return phi.detach().cpu().numpy()

class RBFNetwork(nn.Module):
    def __init__(self, input_dim, num_centers, output_dim):
        super(RBFNetwork, self).__init__()
        self.input_dim = input_dim
        self.num_centers = num_centers
        self.output_dim = output_dim
        
        # 高斯布点
        # # Initialize centers randomly 
        # self.centers = nn.Parameter(torch.randn(num_centers, input_dim))
        
        # # Initialize widths (beta) for each RBF
        # self.beta = nn.Parameter(torch.ones(num_centers)/model['l'])
        
        # # Initialize output weights
        # self.weights = nn.Parameter(torch.randn(num_centers, output_dim))
        
        # # Initialize bias
        # self.bias = nn.Parameter(torch.zeros(output_dim))
   
        self.linear = torch.nn.Linear(num_centers, 1, bias=False)  # 设置bias=False        


        torch.nn.init.normal_(self.linear.weight, mean=0, std=0.1)

   
        # 均匀布点
        # Calculate number of points per dimension to achieve total num_centers
        points_per_dim = int(np.ceil(num_centers ** (1/input_dim)))
        
        # Generate grid points for each dimension
        grid_points = []
        for i in range(input_dim):
            grid_points.append(torch.linspace(-1, 1, points_per_dim))
        
        # Create meshgrid
        mesh = torch.meshgrid(grid_points, indexing='ij')
        
        # Stack and reshape to get all combinations
        centers = torch.stack(mesh, dim=-1).reshape(-1, input_dim)
        
        # If we generated more points than needed, randomly select num_centers points
        if centers.shape[0] > num_centers:
            indices = torch.randperm(centers.shape[0])[:num_centers]
            centers = centers[indices]
            
        self.centers = nn.Parameter(centers, requires_grad=True)  # Make centers non-trainable
        
        # Initialize widths (beta) for each RBF
        self.beta = nn.Parameter(torch.ones(num_centers)/model['l'])
        
        # Initialize output weights
        self.weights = nn.Parameter(torch.randn(num_centers, output_dim))
        
        # Initialize bias
        self.bias = nn.Parameter(torch.zeros(output_dim))
    def forward(self, x):
        # Calculate distances between input and centers
        x_expanded = x.unsqueeze(1)  # [batch_size, 1, input_dim]
        centers_expanded = self.centers.unsqueeze(0)  # [1, num_centers, input_dim]
        distances = torch.sum((x_expanded - centers_expanded) ** 2, dim=2)  # [batch_size, num_centers]
        
        # Calculate RBF activations
        rbf = torch.exp(-self.beta * distances)  # [batch_size, num_centers]
        sum_rbf = torch.sum(rbf, axis = 1).unsqueeze(1)
        output = self.linear(rbf)/sum_rbf
        
        return output

def freeze_layers(model: nn.Module, train_layer_keywords: list, lr=1e-3):
    """
    冻结 model 中除含有指定关键字的层以外的所有参数
    
    参数:
        model (nn.Module): 你的 PyTorch 模型
        train_layer_keywords (list): 需要更新(微调)的层名称关键字的列表。
            比如 ["linear3", "linear4"]，表示只更新名中含 "linear3" 或 "linear4" 的层。
        lr (float): 优化器学习率

    """
    # 1) 遍历所有参数，判断其所属层名字是否包含在指定的层名称关键字
    for name, param in model.named_parameters():
        # 如果关键词命中，就保持 requires_grad=True，否则 requires_grad=False
        if any(keyword in name for keyword in train_layer_keywords):
            param.requires_grad = True
        else:
            param.requires_grad = False
    
def activate_layers(model: nn.Module):
    """
    冻结 model 中除含有指定关键字的层以外的所有参数
    
    参数:
        model (nn.Module): 你的 PyTorch 模型
        train_layer_keywords (list): 需要更新(微调)的层名称关键字的列表。
            比如 ["linear3", "linear4"]，表示只更新名中含 "linear3" 或 "linear4" 的层。
        lr (float): 优化器学习率

    """
    # 1) 遍历所有参数，判断其所属层名字是否包含在指定的层名称关键字
    for name, param in model.named_parameters():
        # 如果关键词命中，就保持 requires_grad=True，否则 requires_grad=False

        param.requires_grad = True





# -----------------------------------------------------------------------------
#  Post-processing helpers for field saving
# -----------------------------------------------------------------------------
def get_material_params(params_file, size, material, gc_scale):
    """Read XDEM material parameters from JSON.

    Supports two JSON structures:
        data[size][material]
        data[material]
    """
    if params_file is None or not Path(params_file).exists():
        print("WARNING: parameter JSON not found. Using default E=210000 MPa, nu=0.3, Gc=2.7.")
        return {"E_MPa": 210.0e3, "nu": 0.3, "Gc_initial_J_m2": 2.7, "l0_norm": 0.015}

    with open(params_file, "r") as f:
        data = json.load(f)

    if size in data and material in data[size]:
        params = data[size][material]
    elif material in data:
        params = data[material]
    else:
        raise KeyError(
            f"Cannot find material={material}, size={size} in {params_file}. "
            f"Top-level keys are: {list(data.keys())}"
        )

    params = dict(params)
    if "E_MPa" not in params:
        if "E_GPa" in params:
            params["E_MPa"] = float(params["E_GPa"]) * 1000.0
        else:
            raise KeyError("JSON parameter set must contain E_MPa or E_GPa.")

    if "Gc_initial_J_m2" not in params:
        if "Gc_initial_from_toughness_J_m2" in params:
            params["Gc_initial_J_m2"] = params["Gc_initial_from_toughness_J_m2"]
        else:
            params["Gc_initial_J_m2"] = 2.7

    params["Gc_used"] = float(params["Gc_initial_J_m2"]) * float(gc_scale)
    params.setdefault("l0_norm", 0.015)
    params.setdefault("nu", 0.3)
    return params


def compute_field_stresses(xGrid, yGrid, u_x, u_y, E_MPa, nu):
    """Compute stress fields from predicted displacement fields.

    Uses the same plane-strain constitutive matrix as DEM_PF.
    Stresses are returned in GPa.
    """
    xvals = xGrid[0, :]
    yvals = yGrid[:, 0]

    # numpy.gradient returns derivatives along axis 0 then axis 1.
    dux_dy, dux_dx = np.gradient(u_x, yvals, xvals, edge_order=2)
    duy_dy, duy_dx = np.gradient(u_y, yvals, xvals, edge_order=2)

    eps_xx = dux_dx
    eps_yy = duy_dy
    gamma_xy = dux_dy + duy_dx

    c11 = E_MPa * (1.0 - nu) / ((1.0 + nu) * (1.0 - 2.0 * nu))
    c22 = c11
    c12 = E_MPa * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
    c21 = c12
    c33 = E_MPa / (2.0 * (1.0 + nu))

    sigma_xx = (c11 * eps_xx + c12 * eps_yy) / 1000.0
    sigma_yy = (c21 * eps_xx + c22 * eps_yy) / 1000.0
    sigma_xy = (c33 * gamma_xy) / 1000.0

    von_mises = np.sqrt(sigma_xx**2 - sigma_xx * sigma_yy + sigma_yy**2 + 3.0 * sigma_xy**2)

    return sigma_xx, sigma_yy, sigma_xy, von_mises


def save_fdgraph_csv(fdGraph, filename="fdGraph.csv"):
    data = fdGraph.copy()
    header = "v_delta,top_edge_average_traction_MPa"
    np.savetxt(filename, data, delimiter=",", header=header, comments="")

# -----------------------------------------------------------------------------
#  Main driver (verbatim structure from original TF script)
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
#  Clean main driver for horizontal edge crack
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="XDEM phase-field simulation for horizontal left-edge crack with top y-loading."
    )

    # Material / JSON inputs
    parser.add_argument("--params", default="xdem_horizontal_edge_outputs/xdem_material_params_by_size_for_xdem.json",
                        help="XDEM material JSON file from extract_horizontal_edge_crack_properties.py.")
    parser.add_argument("--size", default="100_100",
                        help="Size key in JSON, for example 100_100, 300_300, 500_500.")
    parser.add_argument("--material", default="Gr_edge_horizontal",
                        help="Material key in JSON, for example Gr_edge_horizontal.")

    parser.add_argument("--gc_scale", type=float, default=1.0,
                        help="Scale factor for fracture energy Gc.")

    # Crack geometry for attached image
    parser.add_argument("--crack_type", choices=["edge_horizontal", "horizontal_edge", "edge", "diagonal", "cross"],
                        default="edge_horizontal",
                        help="Initial crack geometry. Use edge_horizontal for the attached image.")
    parser.add_argument("--crack_fraction", type=float, default=0.50,
                        help="Normalized horizontal edge-crack length. Attached image uses 0.50.")
    parser.add_argument("--crack_y", type=float, default=0.50,
                        help="Normalized y-position of horizontal crack. Attached image uses 0.50.")
    parser.add_argument("--crack_x_start", type=float, default=0.0,
                        help="Normalized x-start of edge crack. Left edge uses 0.0.")
    parser.add_argument("--crack_tip_radius", type=float, default=0.0,
                        help="Optional normalized crack-tip radius. Usually keep 0.0 for phase-field/XDEM.")
    parser.add_argument("--crack_width", type=float, default=None,
                        help="Normalized crack width. If not given, l0_norm from JSON is used.")

    # Backward-compatible diagonal/cross parameters
    parser.add_argument("--crack_angle", type=float, default=0.0,
                        help="Only used for diagonal/cross cracks.")
    parser.add_argument("--second_crack_angle", type=float, default=None,
                        help="Only used for cross crack. Default is -crack_angle.")

    # Run control
    parser.add_argument("--nsteps", type=int, default=80,
                        help="Number of loading steps.")
    parser.add_argument("--deltaV", type=float, default=0.001,
                        help="Incremental top y-displacement/strain per step.")
    parser.add_argument("--train_points", type=int, default=6400,
                        help="Number of integration/training grid points.")
    parser.add_argument("--grid_points", type=int, default=80,
                        help="Prediction grid points per axis.")
    parser.add_argument("--rbf_centers", type=int, default=500,
                        help="Number of RBF centers for phi network.")
    parser.add_argument("--first_iters", type=int, default=2000,
                        help="Adam iterations for first loading step.")
    parser.add_argument("--later_iters", type=int, default=600,
                        help="Adam iterations for later loading steps.")
    parser.add_argument("--plot_every", type=int, default=1,
                        help="Plot every N loading steps. Use 0 to disable plots.")
    parser.add_argument("--save_model_every", type=int, default=10,
                        help="Save model every N steps. Use 0 to disable model saving.")
    parser.add_argument("--save_fields", action="store_true",
                        help="Save raw XDEM/FEM fields as NPZ files.")
    parser.add_argument("--output_dir", default=None,
                        help="Custom output folder. If not given, an automatic folder is used.")
    parser.add_argument("--force_cpu", action="store_true",
                        help="Force CPU even if CUDA is available.")

    args = parser.parse_args()

    if args.force_cpu:
        device = torch.device("cpu")
        print("Forcing CPU mode")

    originalDir = os.getcwd()

    # ------------------------------------------------------------------
    # Read material parameters
    # ------------------------------------------------------------------
    params = get_material_params(args.params, args.size, args.material, args.gc_scale)

    E_MPa = float(params["E_MPa"])
    nu = float(params["nu"])
    Gc = float(params["Gc_used"])
    l0 = float(params.get("l0_norm", 0.015))
    crack_width = l0 if args.crack_width is None else float(args.crack_width)

    print("====================================")
    print("Running XDEM horizontal left-edge crack")
    print("Material:", args.material)
    print("Size:", args.size)
    print("E_MPa:", E_MPa)
    print("nu:", nu)
    print("Gc used:", Gc)
    print("l0:", l0)
    print("crack_type:", args.crack_type)
    print("crack_fraction:", args.crack_fraction)
    print("crack_y:", args.crack_y)
    print("crack_x_start:", args.crack_x_start)
    print("crack_tip_radius:", args.crack_tip_radius)
    print("crack_width:", crack_width)
    print("nsteps:", args.nsteps)
    print("deltaV:", args.deltaV)
    print("train_points:", args.train_points)
    print("grid_points:", args.grid_points)
    print("rbf_centers:", args.rbf_centers)
    print("first_iters:", args.first_iters)
    print("later_iters:", args.later_iters)
    print("device:", device)
    print("====================================")

    if args.output_dir is None:
        foldername = f"../../results/Tension_2D_{args.material}_{args.size}_horizontal_edge/"
    else:
        foldername = args.output_dir

    createFolder(foldername)

    figHeight = 5
    figWidth = 5
    nSteps = int(args.nsteps)

    # ------------------------------------------------------------------
    # Model dictionary
    # ------------------------------------------------------------------
    second_angle = -float(args.crack_angle) if args.second_crack_angle is None else float(args.second_crack_angle)

    model = {
        "E": E_MPa,
        "nu": nu,
        "Gc": Gc,
        "L": 1.0,
        "W": 1.0,
        "l": l0,
        "lb": np.array([0.0, 0.0], dtype=np.float32),
        "ub": np.array([1.0, 1.0], dtype=np.float32),

        # Horizontal edge crack geometry for attached image
        "crack_type": args.crack_type,
        "crack_fraction": float(args.crack_fraction),
        "crack_y": float(args.crack_y),
        "crack_x_start": float(args.crack_x_start),
        "crack_tip_radius_norm": float(args.crack_tip_radius),
        "crack_width_norm": crack_width,

        # Backward compatibility for diagonal/cross
        "crack_angle_deg": float(args.crack_angle),
        "second_crack_angle_deg": second_angle,
        "crack_center": [0.5, 0.5],

        "B": 1000.0,
    }

    # ------------------------------------------------------------------
    # Generate uniform training/integration points
    # ------------------------------------------------------------------
    n_train_side = int(np.sqrt(args.train_points))
    if n_train_side < 3:
        raise ValueError("train_points is too small. Use at least 9.")

    x = np.linspace(0, 1, n_train_side)
    y = np.linspace(0, 1, n_train_side)
    xx, yy = np.meshgrid(x, y)
    points = np.column_stack((xx.ravel(), yy.ravel()))

    tri = Delaunay(points)
    centroids = np.mean(points[tri.simplices], axis=1)

    # Triangle areas without np.cross deprecation warning for 2D vectors
    v1 = points[tri.simplices][:, 1] - points[tri.simplices][:, 0]
    v2 = points[tri.simplices][:, 2] - points[tri.simplices][:, 0]
    areas = np.abs(v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]) / 2.0

    X_f = np.hstack((centroids, areas.reshape(-1, 1)))
    hist_f = np.zeros((X_f.shape[0], 1), dtype=np.float32)

    # Move into result folder
    os.chdir(os.path.join(originalDir, "./" + foldername + "/"))

    scatterPlot(X_f, figHeight, figWidth, "Training_scatter")
    plt.close("all")

    # ------------------------------------------------------------------
    # Top-edge points for force-displacement response
    # ------------------------------------------------------------------
    N_b = 800
    x_topEdge = np.linspace(0.0, model["L"], N_b, dtype=np.float32)[:, None]
    y_topEdge = np.ones((N_b, 1), dtype=np.float32)
    xTopEdge = np.concatenate([x_topEdge, y_topEdge], axis=1)

    # ------------------------------------------------------------------
    # Prediction grid
    # ------------------------------------------------------------------
    n_pred = int(args.grid_points)
    if n_pred < 3:
        raise ValueError("grid_points is too small. Use at least 3.")

    x = np.linspace(0, 1, n_pred)
    y = np.linspace(0, 1, n_pred)
    xx, yy = np.meshgrid(x, y)
    Grid = np.column_stack((xx.ravel(), yy.ravel()))
    xGrid = xx
    yGrid = yy
    hist_grid = np.zeros((Grid.shape[0], 1), dtype=np.float32)

    scatterPlot(Grid, figHeight, figWidth, "Prediction_scatter")
    plt.close("all")

    fdGraph = np.zeros((nSteps + 1, 2), dtype=np.float32)
    phi_pred_old = hist_grid.copy()

    # ------------------------------------------------------------------
    # Instantiate networks
    # ------------------------------------------------------------------
    modelNN_U_lora = KANLoRA(
        [2, 5, 5, 5, 2],
        base_activation=torch.nn.SiLU,
        grid_size=15,
        grid_range=[0, 1.0],
        spline_order=3,
        lora_rank_base=1,
        lora_rank_spline=1,
    ).to(device)

    activate_layers(modelNN_U_lora)

    modelNN_phi = RBFNetwork(2, int(args.rbf_centers), 1).to(device)
    modelNN = DEM_PF(model, modelNN_U_lora, modelNN_phi).to(device)

    # Initial history field at integration points
    X_f_t = torch.as_tensor(X_f[:, :2], dtype=torch.float32, device=device)
    hist_f = modelNN.net_hist(X_f_t[:, 0:1], X_f_t[:, 1:2]).detach().cpu().numpy()

    # Initial history field at prediction grid points
    Grid_t = torch.as_tensor(Grid[:, :2], dtype=torch.float32, device=device)
    hist_grid_initial = modelNN.net_hist(Grid_t[:, 0:1], Grid_t[:, 1:2]).detach().cpu().numpy()
    hist_grid = hist_grid_initial.copy()

    createFolder("./fields_npz/")
    createFolder("./model/")

    # ------------------------------------------------------------------
    # Loading loop
    # ------------------------------------------------------------------
    v_delta = 0.0

    for iStep in range(1, nSteps + 1):
        num_train_its = int(args.first_iters) if iStep == 1 else int(args.later_iters)

        deltaV = float(args.deltaV)
        hist_deal = "update"
        v_delta += deltaV

        if iStep != 1:
            freeze_layers(
                modelNN_U_lora,
                train_layer_keywords=[
                    "0.base_weight_lora", "0.spline_weight_lora",
                    "1.base_weight_lora", "1.spline_weight_lora",
                    "2.base_weight_lora", "2.spline_weight_lora",
                    "3.base_weight_lora", "3.spline_weight_lora",
                ],
                lr=0.001,
            )

        start = time.time()
        modelNN.train_model(X_f, v_delta, hist_f, num_train_its, hist_deal)
        _, _, phi_f, _, _, hist_f = modelNN.predict(X_f[:, 0:2], hist_f, v_delta)
        elapsed = time.time() - start
        print(f"Training time: {elapsed:.4f}")

        u_pred, v_pred, phi_pred, elas_energy_pred, frac_energy_pred, hist_grid = \
            modelNN.predict(Grid, hist_grid, v_delta)

        phi_pred = np.maximum(phi_pred, phi_pred_old)
        phi_pred_old = phi_pred.copy()

        # Reshape raw fields for saving and post-processing
        u_x_grid = u_pred.reshape(xGrid.shape)
        u_y_grid = v_pred.reshape(xGrid.shape)
        phi_grid = phi_pred.reshape(xGrid.shape)
        elastic_energy_grid = elas_energy_pred.reshape(xGrid.shape)
        fracture_energy_grid = frac_energy_pred.reshape(xGrid.shape)
        hist_grid_2d = hist_grid.reshape(xGrid.shape)

        sigma_xx_GPa, sigma_yy_GPa, sigma_xy_GPa, von_mises_GPa = compute_field_stresses(
            xGrid, yGrid, u_x_grid, u_y_grid, E_MPa, nu
        )

        # ============================================================
        # Save raw XDEM/FEM fields for high-quality post-processing
        # ============================================================
        if args.save_fields:
            np.savez(
                f"./fields_npz/fields_step_{iStep:03d}.npz",
                xGrid=xGrid,
                yGrid=yGrid,
                u_x=u_x_grid,
                u_y=u_y_grid,
                phi=phi_grid,
                elastic_energy=elastic_energy_grid,
                fracture_energy=fracture_energy_grid,
                hist=hist_grid_2d,
                sigma_xx_GPa=sigma_xx_GPa,
                sigma_yy_GPa=sigma_yy_GPa,
                sigma_xy_GPa=sigma_xy_GPa,
                von_mises_GPa=von_mises_GPa,
                v_delta=v_delta,
                E_MPa=E_MPa,
                nu=nu,
                Gc=Gc,
                crack_type=args.crack_type,
                crack_fraction=args.crack_fraction,
                crack_y=args.crack_y,
                crack_x_start=args.crack_x_start,
                crack_tip_radius=args.crack_tip_radius,
            )

        if args.plot_every > 0 and (iStep % int(args.plot_every) == 0 or iStep == 1 or iStep == nSteps):
            fname = str(iStep)
            plotPhiStrainEnerg_uni(xGrid, yGrid, phi_pred, frac_energy_pred, hist_grid, fname, figHeight, figWidth)
            plotDispStrainEnerg_uni(xGrid, yGrid, u_pred, v_pred, elas_energy_pred, fname, figHeight, figWidth)

            adam_buff = modelNN.loss_adam_buff
            lbfgs_buff = np.array(modelNN.lbfgs_buffer)
            plotConvergence(num_train_its, adam_buff, lbfgs_buff, iStep, figHeight, figWidth)
            plt.close("all")

        # Force-displacement response at top edge y=1
        traction_pred = modelNN.predict_traction(xTopEdge, v_delta)
        traction_np = np.asarray(traction_pred).reshape(-1)
        force_y = np.mean(traction_np)

        fdGraph[iStep, 0] = v_delta
        fdGraph[iStep, 1] = float(force_y)

        np.save("./fdGraph.npy", fdGraph)
        save_fdgraph_csv(fdGraph, "./fdGraph.csv")

        print("Relative error phi: skipped for horizontal edge crack")
        print(f"Completed {iStep} of {nSteps}.")

        # Save model only occasionally to avoid disk and memory overhead.
        if args.save_model_every > 0 and (iStep % int(args.save_model_every) == 0 or iStep == nSteps):
            torch.save(modelNN.modelNN_U.state_dict(), "./model/modelNN_U_step_" + str(iStep) + ".pth")
            torch.save(modelNN.modelNN_phi.state_dict(), "./model/modelNN_phi_step_" + str(iStep) + ".pth")

            centers = modelNN.modelNN_phi.centers.detach().cpu().numpy()
            widths = modelNN.modelNN_phi.beta.detach().cpu().numpy()
            np.save("./model/centers_step_" + str(iStep) + ".npy", centers)
            np.save("./model/widths_step_" + str(iStep) + ".npy", widths)

            plt.figure()
            plt.scatter(centers[:, 0], centers[:, 1], c="blue", marker="o", s=5)
            plt.title("RBF Network Centers")
            plt.xlabel("x")
            plt.ylabel("y")
            plt.tight_layout()
            plt.savefig("./model/centers_step_" + str(iStep) + ".pdf", dpi=300)
            plt.close()

        # Memory cleanup
        plt.close("all")
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    plotForceDisp(fdGraph, figHeight, figWidth)
    np.save("./fdGraph.npy", fdGraph)
    save_fdgraph_csv(fdGraph, "./fdGraph.csv")

    os.chdir(originalDir)
    print("Done.")
    print("Result folder:", foldername)

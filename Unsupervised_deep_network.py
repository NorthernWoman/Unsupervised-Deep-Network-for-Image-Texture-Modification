import torch
import numpy as np
import cv2
import imageio
import matplotlib.pyplot as plt
import skimage 
from skimage import io 
from skimage.io import imread
from skimage.io import imsave
from skimage.transform import warp
import warnings
from torchsummary import summary
from skimage import segmentation
from moduls.conv_layers import *
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark =True
dtype = torch.cuda.FloatTensor
warnings.filterwarnings("ignore")
# Input the first image
image = io.imread('./uv23/0.png')
# Input the second image
image0 = io.imread('./uv23/1.png')
# Input the image with high heterogeneous texture
image00 = io.imread('./100.png')
images_warp_np = np.array(image).reshape(-1,416,416)
images_warp_np = np.array(images_warp_np/255,dtype = np.float32)
# The Gaussian Fourier Feature Transform function
dim, nr, nc = 1,416,416
class GaussianFourierFeatureTransform_B(torch.nn.Module):

    def __init__(self, num_input_channels, B, mapping_size=256, scale=10):
        super().__init__()

        self._num_input_channels = num_input_channels
        self._mapping_size = mapping_size
        self._B = B*scale

    def forward(self, x):
        assert x.dim() == 4, 'Expected 4D input (got {}D input)'.format(x.dim())

        batches, channels, width, height = x.shape

        assert channels == self._num_input_channels,\
        "Expected input to have {} channels (got {} channels)".format(self._num_input_channels, channels)

        x = x.permute(0, 2, 3, 1).reshape(batches * width * height, channels)

        x = x @ self._B.to(x.device)

        x = x.view(batches, width, height, self._mapping_size)

        x = x.permute(0, 3, 1, 2)

        x = 2 * np.pi * x
        return torch.cat([torch.sin(x), torch.cos(x)], dim=1)
xy_grid_batch = []
coords_x = np.linspace(-1, 1, nc)
coords_y = np.linspace(-1, 1, nr)
xy_grid = np.stack(np.meshgrid(coords_x, coords_y), -1)
xy_grid_var = np_to_torch(xy_grid.transpose(2,0,1)).type(dtype).cuda()
xy_grid_batch_var = xy_grid_var.repeat(1, 1, 1, 1)
# The Image Generator
model_imgen = conv_layers(832,1)
model_imgen = model_imgen.type(dtype)
summary(model_imgen,(832,416,416))
torch.manual_seed(0)
# BANDWIDTH RELATED SCALE FACTOR
FB_img = 8
B_var = torch.randn(2,416)
# The Grid Generator
pad = 'reflection'
model_grid = conv_layers(2,2, need_sigmoid = False, need_tanh = True).to(device)
sum1 = summary(model_grid,(2, 416, 416))
vec_scale = 1.1
img_gt_batch_var = torch.from_numpy(images_warp_np).type(dtype).cuda()
# The straight grid
grid_input_single_gd = xy_grid_var.detach().clone()
grid_input_gd = xy_grid_batch_var.detach().clone()
# The Gaussian Fourier Feature Transform of straight grid
straight_grid_input = GaussianFourierFeatureTransform_B(2, B_var, 416, FB_img)(xy_grid_batch_var)
grid_input = straight_grid_input.detach().clone()
model_params_list = [{'params':model_grid.parameters()}]
model_params_list.append({'params':model_imgen.parameters()})
# The first stage of learning (formula 2 in paper Mangileva et all 2022)
optimizer = torch.optim.Adam(model_params_list, lr=1e-4)
number = 1000
num_iter_i = number
for epoch in range(num_iter_i):    
    optimizer.zero_grad()
    vec_input = grid_input_single_gd
    refined_xy = [model_grid(vec_input)]
    refined_xy = vec_scale*torch.cat(refined_xy)              
    generated = model_imgen(grid_input)  
    loss = torch.nn.functional.l1_loss(img_gt_batch_var, generated)
    loss += torch.nn.functional.l1_loss(xy_grid_batch_var, refined_xy)    
    loss.backward()
    optimizer.step()   
    if epoch % 50 == 0:
        print('Epoch %d, loss = %.03f' % (epoch, float(loss))) 
# The second stage of learning (formula 3 in paper Mangileva et all 2022)                
images_warp_np = np.array(image0).reshape(-1,416,416)
images_warp_np = np.array(images_warp_np/255,dtype = np.float32)
img_gt_batch_var = torch.from_numpy(images_warp_np).type(dtype).cuda()
num_iter_i = 501
grid = []
losses = []
for epoch in range(num_iter_i):    
    optimizer.zero_grad() 
    vec_input = grid_input_single_gd
    refined_xy = [model_grid(vec_input)]
    refined_xy = vec_scale*torch.cat(refined_xy)              
    generated = model_imgen(GaussianFourierFeatureTransform_B(2, B_var, 416, FB_img)(refined_xy)) 
    loss = torch.nn.functional.l1_loss(img_gt_batch_var, generated)    
    loss.backward()
    optimizer.step()   
    grid.append(refined_xy)
    losses.append(round(float(loss),9))
# Saving the best grid
ind = losses.index(min(losses))
torch.save(grid[ind],'./tensor_23.pt')   
# The re-learning for produce the image with high heterogeneous texture and obtaining the first modified image 
# The formula 4 in paper Mangileva et all 2022                  
images_warp_np = np.array(image00).reshape(-1,416,416)
images_warp_np = np.array(images_warp_np/255,dtype = np.float32)
img_gt_batch_var = torch.from_numpy(images_warp_np).type(dtype).cuda()
num_iter_i = 101
for epoch in range(num_iter_i):    
    optimizer.zero_grad()
    vec_input = grid_input_single_gd
    refined_xy = [model_grid(vec_input)]
    refined_xy = vec_scale*torch.cat(refined_xy)              
    generated = model_imgen(grid_input)  
    loss = torch.nn.functional.l1_loss(img_gt_batch_var, generated)    
    loss.backward()
    optimizer.step()   
    if epoch % 100 == 0:
        print('Epoch %d, loss = %.03f' % (epoch, float(loss)))             
        out_img = generated[0].detach().cpu().numpy().transpose(1,2,0)
        io.imsave('./uv23/0/0.png',out_img[:,:,0])
# Generating the second modified image (formula 5 in paper Mangileva et all 2022)
refined_xy = torch.load('./tensor_23.pt')             
generated = model_imgen(GaussianFourierFeatureTransform_B(2, B_var, 416, FB_img)(refined_xy))
out_img = generated[0].detach().cpu().numpy().transpose(1,2,0)
io.imsave('./uv23/0/1.png',out_img[:,:,0])


    


import argparse
import torch
import os
import numpy as np
import utils
import skimage.color as sc
import cv2
from model import architecture1 as architecture

def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)

# Testing settings
Set5 = r'Test_Datasets/Set5'
Set5_LR = r'Test_Datasets/Set5_LR/x4'
Set5_out = r'results/Set5'
Set14 = r'Test_Datasets/Set14/'
Set14_LR = r'Test_Datasets/Set14_LR/x4/'
Set14_out = r'results/Set14'
B100 = r'Test_Datasets/B100'
B100_LR = r'Test_Datasets/B100_LR/x4'
B100_out = r'results/B100'
Urban100 = r'Test_Datasets/Urban100/'
Urban100_LR = r'Test_Datasets/Urban100_LR/x4/'
Urban100_out = r'results/Urban100'
DIV = r'Test_Datasets/DIV2K/'
DIV_LR = r'Test_Datasets/DIV2K_LR/x4/'
DIV_out = r'results/Urban100'

module_number = 1
K = 2
parser = argparse.ArgumentParser(description='CRFN')
parser.add_argument("--test_hr_folder", type=str, default=DIV,
                    help='the folder of the target images')
parser.add_argument("--test_lr_folder", type=str, default=DIV_LR,
                    help='the folder of the input images')
parser.add_argument("--output_folder", type=str, default=DIV_out)
parser.add_argument("--checkpoint", type=str, default='temp/epoch_160.pth',
                    help='checkpoint folder to use')
parser.add_argument('--cuda', action='store_true', default=True,
                    help='use cuda')
parser.add_argument("--upscale_factor", type=int, default=4,
                    help='upscaling factor')
parser.add_argument("--is_y", action='store_true', default=True,
                    help='evaluate on y channel, if False evaluate on RGB channels')
parser.add_argument("--ext", type=str, default='png',
                    help='file ext')
opt = parser.parse_args()
print(opt)
cuda = opt.cuda
device = torch.device('cuda' if cuda else 'cpu')

filepath = opt.test_hr_folder
ext = opt.ext

if opt.test_hr_folder.endswith("Set5"):
    ext = "bmp"

filelist = utils.get_list(filepath, ext=ext)
lrfilelist = utils.get_list(opt.test_lr_folder, ext=ext)
psnr_list = np.zeros(len(filelist))
ssim_list = np.zeros(len(filelist))
time_list = np.zeros(len(filelist))

model = architecture.CRFN(num_modules=module_number,upscale=opt.upscale_factor,k=K)
print_network(model)

model_dict = utils.load_state_dict(opt.checkpoint)
model.load_state_dict(model_dict, strict=True)

i = 0
start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)


if __name__ == "__main__":
    for x in range(len(filelist)):
        print(x)
        imname = filelist[x]
        im_lrname = lrfilelist[x]
        #print("HR name:",imname)
        #print("LR name:",im_lrname)
        im_gt = cv2.imread(imname, cv2.IMREAD_COLOR)[:, :, [2, 1, 0]]  # BGR to RGB
        im_gt = utils.modcrop(im_gt, opt.upscale_factor)
        im_l = cv2.imread(im_lrname, cv2.IMREAD_COLOR)[:, :, [2, 1, 0]]  # BGR to RGB
        if len(im_gt.shape) < 3:
            im_gt = im_gt[..., np.newaxis]
            im_gt = np.concatenate([im_gt] * 3, 2)
            im_l = im_l[..., np.newaxis]
            im_l = np.concatenate([im_l] * 3, 2)
        im_input = im_l / 255.0
        im_input = np.transpose(im_input, (2, 0, 1))
        im_input = im_input[np.newaxis, ...]
        im_input = torch.from_numpy(im_input).float()

        if cuda:
            model = model.to(device)
            im_input = im_input.to(device)

        with torch.no_grad():
            start.record()
            out = model(im_input)
            end.record()
            torch.cuda.synchronize()
            time_list[i] = start.elapsed_time(end)  # milliseconds

        out_img = utils.tensor2np(out.detach()[0])
        crop_size = opt.upscale_factor
        cropped_sr_img = utils.shave(out_img, crop_size)
        cropped_gt_img = utils.shave(im_gt, crop_size)
        if opt.is_y is True:
            im_label = utils.quantize(sc.rgb2ycbcr(cropped_gt_img)[:, :, 0])
            im_pre = utils.quantize(sc.rgb2ycbcr(cropped_sr_img)[:, :, 0])
        else:
            im_label = cropped_gt_img
            im_pre = cropped_sr_img
        psnr_list[i] = utils.compute_psnr(im_pre, im_label)
        ssim_list[i] = utils.compute_ssim(im_pre, im_label)
        print("PSNR:",psnr_list[i])


        output_folder = os.path.join(opt.output_folder,
                                     imname.split('/')[-1].split('.')[0] + 'x' + str(opt.upscale_factor) + '.png')

        if not os.path.exists(opt.output_folder):
            os.makedirs(opt.output_folder)

        cv2.imwrite(output_folder, out_img[:, :, [2, 1, 0]])
        i += 1


    print("Mean PSNR: {}, SSIM: {}, TIME: {} ms".format(np.mean(psnr_list), np.mean(ssim_list), np.mean(time_list)))

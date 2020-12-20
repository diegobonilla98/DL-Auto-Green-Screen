import cv2
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torchvision.transforms as transforms

import sys

sys.path.append('../../../PycharmProjects/MODNet')
from src.models.modnet import MODNet
import os
import glob


def image_stats(image):
    (l, a, b) = cv2.split(image)
    (lMean, lStd) = (l.mean(), l.std())
    (aMean, aStd) = (a.mean(), a.std())
    (bMean, bStd) = (b.mean(), b.std())
    return lMean, lStd, aMean, aStd, bMean, bStd


def color_transfer(source, target):
    source = cv2.cvtColor(source[:, :, ::-1], cv2.COLOR_BGR2LAB).astype("float32")
    target = cv2.cvtColor(target[:, :, ::-1], cv2.COLOR_BGR2LAB).astype("float32")
    (lMeanSrc, lStdSrc, aMeanSrc, aStdSrc, bMeanSrc, bStdSrc) = image_stats(source)
    (lMeanTar, lStdTar, aMeanTar, aStdTar, bMeanTar, bStdTar) = image_stats(target)
    (l, a, b) = cv2.split(target)
    l -= lMeanTar
    a -= aMeanTar
    b -= bMeanTar
    l = (lStdTar / lStdSrc) * l
    a = (aStdTar / aStdSrc) * a
    b = (bStdTar / bStdSrc) * b
    l += lMeanSrc
    a += aMeanSrc
    b += bMeanSrc
    l = np.clip(l, 0, 255)
    a = np.clip(a, 0, 255)
    b = np.clip(b, 0, 255)
    transfer = cv2.merge([l, a, b])
    transfer = cv2.cvtColor(transfer.astype("uint8"), cv2.COLOR_LAB2BGR)[:, :, ::-1]
    return transfer


backgrounds_idx = 0
backgrounds = [cv2.resize(cv2.imread(i), (672, 512))[:, :, ::-1] for i in glob.glob('./backgrounds/*')] + [None]

torch_transforms = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
)

pretrained_ckpt = '/media/bonilla/HDD_2TB_basura/models/MODNet/modnet_webcam_portrait_matting.ckpt'
modnet = MODNet(backbone_pretrained=False)
modnet = nn.DataParallel(modnet)

modnet = modnet.cuda()
modnet.load_state_dict(torch.load(pretrained_ckpt))
modnet.eval()

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

while True:
    _, frame_np = cap.read()
    frame_np = cv2.cvtColor(frame_np, cv2.COLOR_BGR2RGB)
    frame_np = cv2.resize(frame_np, (910, 512), cv2.INTER_AREA)
    frame_np = frame_np[:, 120:792, :]
    frame_np = cv2.flip(frame_np, 1)

    frame_PIL = Image.fromarray(frame_np)
    frame_tensor = torch_transforms(frame_PIL)
    frame_tensor = frame_tensor[None, :, :, :]
    frame_tensor = frame_tensor.cuda()

    with torch.no_grad():
        _, _, matte_tensor = modnet(frame_tensor, True)

    matte_tensor = matte_tensor.repeat(1, 3, 1, 1)
    matte_np = matte_tensor[0].data.cpu().numpy().transpose(1, 2, 0)
    if backgrounds_idx == len(backgrounds) - 1:
        fg_np = matte_np * frame_np + (1 - matte_np) * cv2.blur(frame_np.copy(), (15, 15))
    else:
        fg_np = matte_np * color_transfer(backgrounds[backgrounds_idx], frame_np.copy()) + (1 - matte_np) * backgrounds[backgrounds_idx]
    view_np = np.uint8(np.concatenate((frame_np, fg_np), axis=1))
    view_np = cv2.cvtColor(view_np, cv2.COLOR_RGB2BGR)

    cv2.imshow('Result', view_np)
    key = cv2.waitKey(1)
    if key == ord('a') & 0xFF:
        backgrounds_idx -= 1
        if backgrounds_idx == -1:
            backgrounds_idx = len(backgrounds) - 1
    elif key == ord('d') & 0xFF:
        backgrounds_idx += 1
        if backgrounds_idx == len(backgrounds):
            backgrounds_idx = 0
    elif key == ord('q') & 0xFF:
        break

cv2.destroyAllWindows()

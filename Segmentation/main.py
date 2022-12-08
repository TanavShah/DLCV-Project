import argparse
import torch
import torch.optim as optim
from torch.autograd import Variable
import cv2
import numpy as np
from skimage import segmentation
import torch.nn.init
from model import SegNet
import os

use_cuda = torch.cuda.is_available()

parser = argparse.ArgumentParser()
parser.add_argument('--input_dir', metavar='FILENAME',
                    help='folder contraining subfolders of images', required=True)
parser.add_argument('--output_dir', metavar='FILENAME',
                    help='output folder where segmentation masks have to saved', required=True)
parser.add_argument('--channels', metavar='N', default=100, type=int, 
                    help='number of channels')
parser.add_argument('--compactness', metavar='C', default=100, type=float, 
                    help='compactness of superpixels')
parser.add_argument('--nConv', metavar='M', default=2, type=int, 
                    help='number of convolutional layers')
parser.add_argument('--minLabels', metavar='minL', default=3, type=int, 
                    help='minimum number of labels')
parser.add_argument('--maxIter', metavar='T', default=1000, type=int, 
                    help='number of maximum iterations')
parser.add_argument('--lr', metavar='LR', default=0.1, type=float, 
                    help='learning rate')
parser.add_argument('--visualize', metavar='1 or 0', default=1, type=int, 
                    help='visualization flag')
parser.add_argument('--num_superpixels', metavar='K', default=10000, type=int, 
                    help='number of superpixels')
args = parser.parse_args()

def slic_feature_extraction(img):
    lab = segmentation.slic(img, compactness=args.compactness, n_segments=args.num_superpixels)
    lab = lab.reshape(img.shape[0]*img.shape[1])
    unique = np.unique(lab)
    slic_feat = []
    for i in range(len(unique)):
        slic_feat.append( np.where(lab == unique[i])[0])

    return slic_feat


def load_img(img_path):
    img = cv2.resize(cv2.imread(img_path), (500,500))
    data = torch.from_numpy(np.array([img.transpose((2, 0, 1)).astype('float32')/255.]))
    if use_cuda: data = data.cuda()
    data = Variable(data)

    return data, img

def train(data,img, slic_feat, out_path):

    model = SegNet( data.size(1), args.channels, args.nConv )
    if use_cuda: model.cuda()

    model.train()

    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
    label_colours = np.random.randint(255,size=(100,3))

    for batch_idx in range(args.maxIter):
        optimizer.zero_grad()
        
        output = model(data)[0]
        output = output.permute(1,2,0).contiguous().view(-1, args.channels)
        ignore, target = torch.max( output, 1 )
        im_target = target.data.cpu().numpy()
        nLabels = len(np.unique(im_target))
        
        if args.visualize:
            out_im = np.array([label_colours[ c % 100 ] for c in im_target])
            out_im = out_im.reshape(img.shape).astype( np.uint8 )
            cv2.imshow( "output", out_im )
            cv2.waitKey(10)

        # improving segmenration using superpixel
        for i in range(len(slic_feat)):
            labels_per_sp = im_target[slic_feat[i]]
            u_labels_per_sp = np.unique( labels_per_sp )
            hist = np.zeros(len(u_labels_per_sp))
            for j in range(len(hist)):
                hist[j] = len( np.where( labels_per_sp == u_labels_per_sp[j])[0] )
            im_target[slic_feat[i]] = u_labels_per_sp[np.argmax(hist)]
        target = torch.from_numpy( im_target )

        if use_cuda: target = target.cuda()

        target = Variable(target)
        loss = loss_fn(output, target)
        loss.backward()
        optimizer.step()

        if batch_idx % 10 == 0: print (batch_idx, '/', args.maxIter, ':', nLabels, loss.item())

        if nLabels <= args.minLabels:
            print ("nLabels", nLabels, "reached minLabels", args.minLabels, ".")
            break

    if not args.visualize:
        output = model(data)[0]
        output = output.permute(1,2,0).contiguous().view( -1, args.channels)
        ignore, target = torch.max(output, 1)
        im_target = target.data.cpu().numpy()
        out_im = np.array([label_colours[c % 100] for c in im_target])
        out_im = out_im.reshape(img.shape).astype(np.uint8)
    cv2.imwrite(out_path, out_im)


def unsuper_Seg(file_path, out_path):

    data, img = load_img(file_path)
    slic_feat = slic_feature_extraction(img)

    train(data, img, slic_feat, out_path)

if __name__ == "main":
    folder = args.input_dir
    out = args.output_dir
    
    for sub_folder in os.listdir(folder):
        sub_path = os.path.join(folder, sub_folder)
        out_folder = os.path.join(out, sub_folder)

        for file in os.listdir(sub_folder):
            file_path = os.path.join(sub_folder, file)
            out_path = os.path.join(out_folder, file)
            print(file_path)

            unsuper_Seg(file_path, out_path)







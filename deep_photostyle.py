import argparse
from PIL import Image
import numpy as np
import os
from photo_style import stylize
import tensorflow as tf
import pandas as pd
from PIL import Image

from vgg19.vgg import Vgg19

parser = argparse.ArgumentParser()
# Input Options
parser.add_argument("--content_image_path", dest='content_image_path',  nargs='?',
                    help="Path to the content image")
parser.add_argument("--style_image_path",   dest='style_image_path',    nargs='?',
                    help="Path to the style image")
parser.add_argument("--content_seg_path",   dest='content_seg_path',    nargs='?',
                    help="Path to the style segmentation")
parser.add_argument("--style_seg_path",     dest='style_seg_path',      nargs='?',
                    help="Path to the style segmentation")
parser.add_argument("--init_image_path",    dest='init_image_path',     nargs='?',
                    help="Path to init image", default="")
parser.add_argument("--output_image",       dest='output_image',        nargs='?',
                    help='Path to output the stylized image', default="best_stylized.png")
parser.add_argument("--serial",             dest='serial',              nargs='?',
                    help='Path to save the serial out_iter_X.png', default='./')

# Training Optimizer Options
parser.add_argument("--max_iter",           dest='max_iter',            nargs='?', type=int,
                    help='maximum image iteration', default=1000)
parser.add_argument("--learning_rate",      dest='learning_rate',       nargs='?', type=float,
                    help='learning rate for adam optimizer', default=1.0)
parser.add_argument("--print_iter",         dest='print_iter',          nargs='?', type=int,
                    help='print loss per iterations', default=1)
# Note the result might not be smooth enough since not applying smooth for temp result
parser.add_argument("--save_iter",          dest='save_iter',           nargs='?', type=int,
                    help='save temporary result per iterations', default=100)
parser.add_argument("--lbfgs",              dest='lbfgs',               nargs='?',
                    help="True=lbfgs, False=Adam", default=False)

# Weight Options
parser.add_argument("--content_weight",     dest='content_weight',      nargs='?', type=float,
                    help="weight of content loss", default=5e0)
parser.add_argument("--style_weight",       dest='style_weight',        nargs='?', type=float,
                    help="weight of style loss", default=1e2)
parser.add_argument("--tv_weight",          dest='tv_weight',           nargs='?', type=float,
                    help="weight of total variational loss", default=1e-3)
parser.add_argument("--affine_weight",      dest='affine_weight',       nargs='?', type=float,
                    help="weight of affine loss", default=1e4)

# Style Options
parser.add_argument("--style_option",       dest='style_option',        nargs='?', type=int,
                    help="0=non-Matting, 1=only Matting, 2=first non-Matting, then Matting", default=0)
parser.add_argument("--apply_smooth",       dest='apply_smooth',        nargs='?',
                    help="if apply local affine smooth", default=False)

# Smoothing Argument
parser.add_argument("--f_radius",           dest='f_radius',            nargs='?', type=int,
                    help="smooth argument", default=15)
parser.add_argument("--f_edge",             dest='f_edge',              nargs='?', type=float,
                    help="smooth argument", default=1e-1)

args = parser.parse_args()


vgg_model = tf.keras.applications.vgg19.VGG19(
    include_top=True,
    weights='imagenet',
    input_tensor=None,
    input_shape=None,
    pooling=None,
    classes=1000,
    classifier_activation='softmax'
)

vgg_model.layers[1]._name = "conv1_1"
vgg_model.layers[2]._name = "conv1_2"
vgg_model.layers[3]._name = 'pool1'

vgg_model.layers[4]._name = "conv2_1"
vgg_model.layers[5]._name = "conv2_2"
vgg_model.layers[6]._name = 'pool2'

vgg_model.layers[7]._name = "conv3_1"
vgg_model.layers[8]._name = "conv3_2"
vgg_model.layers[9]._name = "conv3_3"
vgg_model.layers[10]._name = "conv3_4"
vgg_model.layers[11]._name = 'pool3'

vgg_model.layers[12]._name = "conv4_1"
vgg_model.layers[13]._name = "conv4_2"
vgg_model.layers[14]._name = "conv4_3"
vgg_model.layers[15]._name = "conv4_4"
vgg_model.layers[16]._name = 'pool4'

vgg_model.layers[17]._name = "conv5_1"
vgg_model.layers[18]._name = "conv5_2"
vgg_model.layers[19]._name = "conv5_3"
vgg_model.layers[20]._name = "conv5_4"
vgg_model.layers[21]._name = 'pool5'

inputs = vgg_model.input
x = inputs

for layer in vgg_model.layers:
    if layer._name == "fc2":
        x = tf.keras.layers.Dense(1024, activation='relu', name='fc2')(x)
    elif layer._name == "predictions":
        x = tf.keras.layers.Dense(3, activation='softmax', name='predictions')(x)
    else:
        x = layer(x)
        layer.trainable = False

vgg_model = tf.keras.Model(inputs, x)

print(vgg_model.summary())

# train_data = pd.read_csv("dummy_data.csv")

x_train = []
y_train = []

# portraits
for i in range(129):
    y_train.append(0)
    image_path = "new_dataset/portraits/" + str(i + 1) + ".jpg"
    img = Image.open(image_path).convert('RGB')
    img = img.resize((224, 224))
    img = np.array(img)
    x_train.append(img)

# landscape
for i in range(135):
    y_train.append(1)
    image_path = "new_dataset/landscape/" + str(i + 1) + ".jpg"
    img = Image.open(image_path).convert('RGB')
    img = img.resize((224, 224))
    img = np.array(img)
    x_train.append(img)

# wildlife
for i in range(136):
    y_train.append(2)
    image_path = "new_dataset/wildlife/" + str(i + 1) + ".jpg"
    img = Image.open(image_path).convert('RGB')
    img = img.resize((224, 224))
    img = np.array(img)
    x_train.append(img)

# for index, row in train_data.iterrows():

#     if index == 10:
#         break

#     y_train.append(row['Label'])
#     image_path = "examples/input/" + row['Image'] + ".jpg"
#     img = Image.open(image_path)
#     img = img.resize((224, 224))
#     img = np.array(img)
#     x_train.append(img)


x_train = np.array(x_train)
y_train = np.array(y_train)

print(x_train.shape, y_train.shape)

vgg_model.compile(optimizer=tf.keras.optimizers.Adam(), loss=tf.keras.losses.SparseCategoricalCrossentropy())
history = vgg_model.fit(x_train, y_train, batch_size=20, epochs=20)

print("Unfreeze the whole model, training again:")

for layer in vgg_model.layers:
    print(layer._name)
    layer.trainable = True

vgg_model.compile(optimizer=tf.keras.optimizers.Adam(), loss=tf.keras.losses.SparseCategoricalCrossentropy())

print(vgg_model.summary())

history = vgg_model.fit(x_train, y_train, batch_size=20, epochs=20)

# Save trained VGG model
vgg_model.save("vgg_model")

# Load trained VGG model
# vgg_model = tf.keras.models.load_model("vgg_model")

parser.add_argument("--vgg_model", dest='vgg_model', nargs='?', type=type(vgg_model), help="Vgg pre-trained model", default=vgg_model)

# def main():
#     if args.style_option == 0:
#         best_image_bgr = stylize(args, False)
#         result = Image.fromarray(np.uint8(np.clip(best_image_bgr[:, :, ::-1], 0, 255.0)))
#         result.save(args.output_image)
#     elif args.style_option == 1:
#         best_image_bgr = stylize(args, True)
#         if not args.apply_smooth:
#             result = Image.fromarray(np.uint8(np.clip(best_image_bgr[:, :, ::-1], 0, 255.0)))
#             result.save(args.output_image)
#         else:
#             # Pycuda runtime incompatible with Tensorflow
#             from smooth_local_affine import smooth_local_affine
#             content_input = np.array(Image.open(args.content_image_path).convert("RGB"), dtype=np.float32)
#             # RGB to BGR
#             content_input = content_input[:, :, ::-1]
#             # H * W * C to C * H * W
#             content_input = content_input.transpose((2, 0, 1))
#             input_ = np.ascontiguousarray(content_input, dtype=np.float32) / 255.

#             _, H, W = np.shape(input_)

#             output_ = np.ascontiguousarray(best_image_bgr.transpose((2, 0, 1)), dtype=np.float32) / 255.
#             best_ = smooth_local_affine(output_, input_, 1e-7, 3, H, W, args.f_radius, args.f_edge).transpose(1, 2, 0)
#             result = Image.fromarray(np.uint8(np.clip(best_ * 255., 0, 255.)))
#             result.save(args.output_image)
#     elif args.style_option == 2:
#         args.max_iter = 2 * args.max_iter
#         tmp_image_bgr = stylize(args, False, vgg_model)
#         result = Image.fromarray(np.uint8(np.clip(tmp_image_bgr[:, :, ::-1], 0, 255.0)))
#         args.init_image_path = os.path.join(args.serial, "tmp_result.png")
#         result.save(args.init_image_path)

#         best_image_bgr = stylize(args, True, vgg_model)
#         if not args.apply_smooth:
#             result = Image.fromarray(np.uint8(np.clip(best_image_bgr[:, :, ::-1], 0, 255.0)))
#             result.save(args.output_image)
#         else:
#             from smooth_local_affine import smooth_local_affine
#             content_input = np.array(Image.open(args.content_image_path).convert("RGB"), dtype=np.float32)
#             # RGB to BGR
#             content_input = content_input[:, :, ::-1]
#             # H * W * C to C * H * W
#             content_input = content_input.transpose((2, 0, 1))
#             input_ = np.ascontiguousarray(content_input, dtype=np.float32) / 255.

#             _, H, W = np.shape(input_)

#             output_ = np.ascontiguousarray(best_image_bgr.transpose((2, 0, 1)), dtype=np.float32) / 255.
#             best_ = smooth_local_affine(output_, input_, 1e-7, 3, H, W, args.f_radius, args.f_edge).transpose(1, 2, 0)
#             result = Image.fromarray(np.uint8(np.clip(best_ * 255., 0, 255.)))
#             result.save(args.output_image)

# if __name__ == "__main__":
#     main()

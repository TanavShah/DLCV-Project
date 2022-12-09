<div align="center">

<samp>

<h1> COMS W 4995 006 (Fall 2022) DLCV - Automated Photo Editing using Neural Style Transfer</h1>

<h3> Adithya K Krishna: akk2188 <br> Tanav Hitendra Shah: ts3469 <br> Rohit V Gopalakrishnan: rvg2119 </h3>
</samp>   

</div>     

## Dataset
For the purpose of our project we have collected our own dataset by scarping from internet and creating the segmentation masks required for the model using an unsupervised segmentation model.
The dataset is available at : https://drive.google.com/drive/folders/14wCyJRV5z9f4AVwjEmQYUqAJofcP19qQ?usp=sharing


## Directory setup
<!---------------------------------------------------------------------------------------------------------------->
The structure of the repository is as follows: 

- `data_utils/web_scrape.py`: Contains code for web-scraping given a link
- `data_utils/segmentation`: Contains the segmentation code for getting semgentation masks for the scraped images.
- `data_demo.ipynb`: python notebook that shows how data is extracted, segmented and saved
-
-
-

---

## Dependencies
* BeautifulSoup
* Pytorch
* Selenium
* [Tensorflow](https://www.tensorflow.org/)
* [Numpy](www.numpy.org/)
* [Pillow](https://pypi.python.org/pypi/Pillow/)
* [Scipy](https://www.scipy.org/)
* [PyCUDA](https://pypi.python.org/pypi/pycuda) (used in smooth local affine, tested on CUDA 8.0)

***It is recommended to use [Anaconda Python](https://www.continuum.io/anaconda-overview), since you only need to install Tensorflow and PyCUDA manually to setup. The CUDA is optional but really recommended***

### Download the VGG-19 model weights
The VGG-19 model of tensorflow is adopted from [VGG Tensorflow](https://github.com/machrisaa/tensorflow-vgg) with few modifications on the class interface. The VGG-19 model weights is stored as .npy file and could be download from [Google Drive](https://drive.google.com/file/d/0BxvKyd83BJjYY01PYi1XQjB5R0E/view?usp=sharing&resourcekey=0-Q2AewV9J7IYVNUDSnwPuCA) or [BaiduYun Pan](https://pan.baidu.com/s/1o9weflK). After downloading, copy the weight file to the **./project/vgg19** directory

## Usage
### Basic Usage
You need to specify the path of content image, style image, content image segmentation, style image segmentation and then run the command

```
python deep_photostyle.py --content_image_path <path_to_content_image> --style_image_path <path_to_style_image> --content_seg_path <path_to_content_segmentation> --style_seg_path <path_to_style_segmentation> --style_option 2
```


## Contact
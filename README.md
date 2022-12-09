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


## Usage
### Basic Usage

Download the data.zip file from the Google drive folder, and unzip its contents. This contains the image data and some other files needed to execute the code.

Create a conda environment with python 3.8.15 and install the dependencies using requirements.txt file. Then run the script using the following command:

```
python edit_image.py
```


## Contact

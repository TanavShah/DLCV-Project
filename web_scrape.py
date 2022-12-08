import time
import requests
import os
import bs4
import requests
from selenium import webdriver
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--foldername', metavar='FILENAME',
                    help='folder contraining images', required=True)
parser.add_argument('--search_URL',
                    help='url to be scraped', required=True)
args = parser.parse_args()

DRIVER_PATH = '/home/adithya/Downloads/chromedriver_linux64/chromedriver'
driver = webdriver.Chrome(executable_path=DRIVER_PATH)

def download_image(url, folder_name, num):
    # write image to file
    reponse = requests.get(url)
    if reponse.status_code==200:
        with open(os.path.join(folder_name, str(num)+".jpg"), 'wb') as file:
            file.write(reponse.content)


def scrape(num_images = 151):
    for i in range(1, num_images):
        if i % 25 == 0:
            # to skip suggestions in goole
            continue

        xPath = """//*[@id="islrg"]/div[1]/div[%s]"""%(i)

        previewImageXPath = """//*[@id="islrg"]/div[1]/div[%s]/a[1]/div[1]/img"""%(i)
        previewImageElement = driver.find_element("xpath", previewImageXPath)
        previewImageURL = previewImageElement.get_attribute("src")

        driver.find_element("xpath", xPath).click()

        timeStarted = time.time()
        while True:
            imageElement = driver.find_element("xpath", """//*[@id="Sva75c"]/div/div/div/div[3]/div[2]/c-wiz/div[2]/div[1]/div[1]/div[2]/div/a/img""")
            imageURL= imageElement.get_attribute('src')

            if imageURL != previewImageURL: break
            else:  
                currentTime = time.time()
                if currentTime - timeStarted > 10:
                    print("Timeout! Will download a lower resolution image and move onto the next one")
                    break
        #Downloading image
        try:
            download_image(imageURL, args.folder_name, i)
            print("Downloaded element %s out of %s total. URL: %s" % (i, num_images, imageURL))
        except:
            print("Couldn't download an image %s, continuing downloading the next one"%(i))

if __name__ == "main":
    folder_name = args.foldername #'./wildlife/'

    search_URL = args.search_URL
    driver.get(search_URL)

    driver.execute_script("window.scrollTo(0, 0);")
    page_html = driver.page_source
    pageSoup = bs4.BeautifulSoup(page_html, 'html.parser')

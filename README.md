[TOC]

## Website UI
![Website UI](test_images/Screenshot%202024-06-04%20013818.png)

# Classificatiom of a footballer based on the picture using Machine learning.
### A fully functional Website that uses photos to classify footballers (Classifier: SVM).

- The development of the model is thoroughly documented in the notebook [[file](Model.ipynb "file")].
- The [server](server "server"), util, and [web page](UI "web page") files document are in this read's relevant section.md file.

#### Data procurement
- Data was procured using the Fatkum batch image download extension [Link.](https://fatkun-batch-download-image.en.softonic.com/chrome/extension "Link.")

#### Data pre-processing, feature engineering, and model training.
- All of this could be found in the notebook.

#### Server and port hosting.
The server was created using the flask module. 

#### Website.
CSS, HTML, and JavaScript files are present in the [UI](UI "client") folder. In the future, this website might be hosted on a server but it will be a better version with much more data.
If the model fails to recognize any face or set of eyes, it looks like this:
![Website UI](test_images/Screenshot%202024-06-04%20013926.png)

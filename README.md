# Photo to Monet Painting Generator using CycleGAN

The goal of this project was to build and train a generative model to convert normal photographs to Monet-style paintings, and then build a web application that allows users to input custom images and view the generated output.

The dataset used for this project was from this Kaggle competition: {}. At the time of writing this documentation this model is ranked 53/106 with a score of {}. The architecture used for the generative model was CycleGAN, proposed by this paper: {} (more details below). 

For the web application, React.js was used for the frontend and a Flask API was used for the backend.

## Demo


## CycleGAN Architecture and Training Details

The CycleGAN model was implemented using Python and Pytroch, and training was done in Google Colab. The model was trained on 600 photos and 250 Monet paintings. The parameters used for training can all be found in the Jupyter Notebook.

## Key Takeaways and Reflection

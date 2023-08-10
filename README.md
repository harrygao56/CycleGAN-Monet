# Photo to Monet Painting Generator using CycleGAN

The goal of this project was to build and train a generative model to convert normal photographs to Monet-style paintings, and then build a web application that allows users to input custom images and view the generated output.

The dataset used for this project was from this Kaggle competition: https://www.kaggle.com/code/harrygao1/cyclegan-monet-generator. At the time of writing this documentation this model is ranked 53/106 with a score of 60.7. The architecture used for the generative model was CycleGAN, proposed by this paper: https://arxiv.org/abs/1703.10593 (more details below). 

For the web application, React.js was used for the frontend and a Flask API was used for the backend.

## Demo

![](https://github.com/harrygao56/CycleGAN-Monet/blob/main/Demo.gif)

## CycleGAN Architecture and Training Details

The CycleGAN model was implemented using Python and Pytorch, and training was done in Google Colab. The model was trained on 600 photos and 250 Monet paintings. The parameters used for training can all be found in the Jupyter Notebook.

## Takeaways and What I Learned

- PyTorch Transforms: for CV models that have a set input size, you can use transforms to resize data
- Major error I made: I only notices this as I was going through my code, but I did the identity loss totally wrong. Rather than id_loss(gen(monet), monet), I did id_loss(gen(photo), photo) which LITERALLY constrains the output to the input. Somehow the results were still decent I have no idea how.
- Before an inputted image is fed through the model, it might need to be unsqueezed to add the extra batch dimension
- Deploying a model to a web app isn't that hard. Just create a Flask API.

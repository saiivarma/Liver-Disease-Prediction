# Liver-Disease-Prediction
Liver disease prediction model using ILPD dataset and Multilayered perceptron and XGBoost

## Dataset Description

1. The .csv file with name Indian Liver Patient Dataset (ILPD) is the original dataset downloaded from the machine learning repositry.

2. The .csv file with name input_data is the cleaned data.

3. The .csv file with name Input_data_normalised contains the normalised and the preprocessed data.


### Multilayered Perceptron with 1 hidden layer.
    Attained 74% accuracy with the sigmoid activation.
    The random state has to be updated

### Multilayered perceptron with 2 hidden layer.
    Attained 71% accuracy with the sigmoid activation.
    The Random_state has to be updated
    
### Multilayered perceptron with 2 hidden layer(keras).
    Attained 74% accuracy with relu activation function.
    Attained 75% accuracy with the tanh activation function.

### XGBoost Model yet to be updated
    Parameters to be tweaked,attained 72 percentage accuracy for default parametres and learning rate 0.05.
    
## Flask
    used flask for the integration of the model with webpage
    used particle.js to create the particle effect.

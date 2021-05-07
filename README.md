## Keras-1D-ACGAN-Data-Augmentation

This code is based on the code from the referenced site.

[Reference](https://machinelearningmastery.com/how-to-develop-an-auxiliary-classifier-gan-ac-gan-from-scratch-with-keras/)

This code would be useful to whom are going to use **(1) an 1-D dataset classification based on the GAN model** or  **(2) 1-D data Augmentation based on the GAN**.


### How to Generate the Sample (Data-augmentation) using this Code

you can get the generated data sinutaneously, as you are running this code.

The generated data is saved to the **.csv** format.

This is worked by this code in the file.
```python
generated_fake_data = np.append(X_fake_temp, labels_fake_temp, axis=1)
np.savetxt('generated_data/generated_fake_data %s th.csv' % (i + 1), generated_fake_data, delimiter=",")
```


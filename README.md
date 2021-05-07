## Keras-1D-ACGAN-Data-Augmentation

This code is based on the code from the referenced site.

[Reference](https://machinelearningmastery.com/how-to-develop-an-auxiliary-classifier-gan-ac-gan-from-scratch-with-keras/)

This code would be useful to whom are going to use **(1) an 1-D dataset classification based on the GAN model** or  **(2) 1-D data Augmentation based on the GAN**.

If running it, you can see a screen like below one. As model is continuosly saved, you could stop as you are satisfied with the results.

![20210508_001941](https://user-images.githubusercontent.com/71545160/117471911-4c4b7900-af93-11eb-8d91-48349d8ec8f4.png)


### How to Generate the Sample (Data-augmentation) using this Code

you can get the generated data sinutaneously, as you are running this code.

The generated data is saved to the **.csv** format.

This is worked by this code in the file.
```python
generated_fake_data = np.append(X_fake_temp, labels_fake_temp, axis=1)
np.savetxt('generated_data/generated_fake_data %s th.csv' % (i + 1), generated_fake_data, delimiter=",")
```

The outputs are saved like below.

![20210508_002017](https://user-images.githubusercontent.com/71545160/117471925-4f466980-af93-11eb-834a-66af833d8717.png)

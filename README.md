# BinaryTernary-MNIST-training

Training of MNIST dataset with float, binary and ternary nets models.

Original code for Binary and Ternary nets: https://github.com/DingKe/nn_playground.git

To train choose one of the yaml configs or create new one. Specific architecture (see models.py), number of layers and neurons, loss function, L1 regularization can be defined in the yaml config. File with pruned weights for retraining and compress the model can also be added.

To train use command:

```
python mnist_mlp.py -c mnist_784x128x128x128x10_float_softmax.yml
```

To evaluate performance:

```
python eval_mnist.py -c mnist_784x128x128x128x10_ternary_maxrelu.yml
```

To convert the data into text files for the HLS test bench add option ```-C``` to the command above. The performance of the HLS test bench output can also be evaluated adding the option ```--checkHLS test_bench_output.dat```



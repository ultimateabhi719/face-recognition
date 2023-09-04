# Siamese Network for Face Recognition


We implement the Siamese Network for face recognition using ResNet-50. To reduce the training time (and data required) we freeze all but the last stage of the ResNet. ( Even the last stage we reduce the number of bottleneck layers to 1--instead of 3. ). We use a contrastive loss for comparing the image representations, obtaining a test accuracy of 82% using the face aligned celebA dataset. We also tested a learned loss function. But it did not provide as good performance!


An example set of parameters is defined in the `src/main.py` file. Most of the parameter names are self-explanatory. 

model_params
	- fc_dim : dimension of fully connected layer in the CNN backbone.
	- out_dim :  dimension of fully connected layer in the CNN backbone.

data_params
	- train_datapath : (training data image root directory, csv containing headers `img`, `ids`)
	- traindata_multfactor : ratio of image pairs to 
	- val_datapath : ...
	- valdata_multfactor : ...
	- test_datapath : ...
	- testdata_multfactor : ...

Note that the Siamese network requires pairs of images as input.  The `traindata_multfactor`, `valdata_multfactor`, and `testdata_multfactor` correspond to the ratio of the number of generated image pairs to the size of the image dataset for training, validation and test sets respectively.

train_params
	- batch_size : training batch size
	- resume_dir : directory for saved model checkpoints to resume training
	- save_prefix : directory to save model checkpoint and logs
	- epochs : number of epochs to train for
	- save_freq : number of batches between saving model checkpoints
	- save_model : format for saving model
	- save_loss : format for saving loss

To train the model specify the parameters in `src/main.py` file, and run the command :  `python src/main.py train`. To evaluate the model on the test set you can use: `python src/main.py train`. 

If you want to optimize the parameters of the model, you can find out the optimal parameters using `python src/main.py optimize`
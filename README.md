# Supplimentary Scripts for Thesis Completion
## Main Scripts:
* Run glow_training.ipynb first to train the encoder and decoder.
* Run jpeg_to_tensor.ipynb to convert test images to tensors, facilitating batch predictions.
* Run encode_single.ipynb with decode_single.ipynb to get a demo prediction & Dice calculation
* Run encode_multi_v1.ipynb to batch encode the test images. It uses a simple loop.
* The other encode_multi_v2.ipynb is supposed to be faster than v1, however is outputting weired results, I will debug it later.
* A decode_multi.ipynb script will be provided soon.

## Auxiliary Scripts:
* Run mat_to_jpg.py in the JMW_Data_mining_brain_Res folder to generate stacked images.
* Run data_formatting.py with partition.txt (my demo_partition.txt only produces 1 image for demo) to convert images to .tfrecord format.
* Put all the rest into the same directly as the Main Scripts.


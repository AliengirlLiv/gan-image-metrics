# gan-image-metrics
GANgsters

To run the segmentation model: Import the file semantic-segmentation-model/test.py.  Load the segmentation model using its function setup_test().  To run the model on a single image, call get_last_hidden(model, img).  An example function test_all() shows this in action.

To run all eval metrics together, run python3 eval.py --real_path "Path/to/real/images" --gen_path "path/to/generated/images" --save_file "optional/filename/to/save/scores.txt"



# gan-image-metrics
GANgsters

To run the segmentation model: Import the file semantic-segmentation-model/test.py.  Load the segmentation model using its function setup_test().  To run the model on a single image, call get_last_hidden(model, img).  An example function test_all() shows this in action.

To get inception scores, generate a bunch of images and place them into a folder.  Get the file improved_gan/inception_score/test_inception.py, and call its function test_inception with a path to the images.

To get FID scores, change the data_path variable in TTUR/precalc_stats_example.py to the path to a set of real images.  Run that script.  Next, change the image_path variable in TTUR/fid_example.py and run that script.  It will print the FID.


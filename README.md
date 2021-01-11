# if-stitch

if-stitch is a Python script for 2D-stitching of images from immunofluorescence microscopy.
Typically, many images are taken from 96- or 384-well microplates. One well can consist of a grid of several fields.
The aim is to stitch the field of particular wells together so that cell in the border area can be mapped completely.
This might be helpful for cell segmentation where truncated cells would be excluded otherwise.

## How it works

<img src="https://https://github.com/marx-alex/if-stitch/blob/master/image_stitching_opencv_pipeline.png" align="center"
     alt="Open CV stitching pipeline" width="401" height="600">
# Senior Project

This repository contains artifacts from the project: **Optimizing the 3D Reconstruction of Indoor Scenes Using 3DGS**.

---

## Repository Details

The script for sampling images from a video input file is `sampling.py`. The script provides two sampling approaches: interval-based and quality-based.  
More details can be found using:

```bash
python sampling.py -h
```

The script for edge-enhancing images is edge_enhancement.py.
More details can be found using:
```bash
python edge_enhancement.py -h
```

The folder `\pruning_method` contains the Gaussian Splatting implementation with compression used in our project. This implementation is based on the NerfStudio framework.

More details about our project can be found in the [demo video](https://youtu.be/u-RoqbRMd14?si=fP5OmtIYsBsPbgxT).

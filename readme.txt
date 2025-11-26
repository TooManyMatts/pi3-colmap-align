pi3-colmap-align/
├── data/
│   ├── colmap/
│   │   └── {dataset_name}/                # e.g. t01_v01...
│   │       ├── colmap_files/              # Sparse reconstruction files
│   │       │   ├── images.txt             # (or .bin)
│   │       │   ├── cameras.txt
│   │       │   └── points3D.txt
│   │       └── images/                    # Source images
│   │           └── image_000001.jpg...
│   └── pi3/
│       └── {dataset_name}/
│           ├── pi3_files/
│           │   ├── results_pie.pt         # Raw tensors
│           │   └── pc_pi3.ply             # Unaligned preview
│           └── images/                    # (Optional/Duplicate of colmap images)
│
├── src/
│   ├── __init__.py
│   ├── loaders.py                         # Parsers for .txt/.bin and .pt
│   ├── geometry.py                        # Umeyama & transformation math
│   └── io.py                              # PLY savers
│
├── output/                                # Saved here
│   └── {dataset_name}/
│       └── aligned_pi3.ply
│
├── align.py                               # Main CLI script
└── requirements.txt

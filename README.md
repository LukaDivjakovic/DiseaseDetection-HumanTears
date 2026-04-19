# DiseaseDetection-HumanTears
HackKosice project for UPJS

## How to run?

1. Create a virtual environment (python version 3.12)
2. Activate the virtual environment
3. Run `pip install -r requirements.txt`
4. Run inference with the trained ResNet50 model:

   ```
   python solution.py <path_to_test_set>
   ```

   Where `<path_to_test_set>` is either:
   - a directory containing `.bmp` images (searched recursively), or
   - a path to a single `.bmp` image.

   The script loads the trained model from `models/resnet50_final.pth` and
   prints one line per image in the format:

   ```
   <picture path>: <predicted class>
   ```
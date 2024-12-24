# Deforestation Detection

## Container
A docker container has been provided that contains all necessary software to train the models:
```
jalexhurt/sc23-deforestation-detection
```

This container was used to train all [ChangeFormer](https://github.com/wgcban/ChangeFormer) models.

## Data Download and Processing

The data processing pipeline consists of sequential steps, where the output of one step serves as the input for the subsequent step.

### Step 1: Downloading Sentinel-2 Tiles 

The following requirements must be met:

1. **Create a Sentinel-2 API Account:** If an account is not already available, create one by visiting the [Copernicus Open Access Hub](https://scihub.copernicus.eu/).

2. **Set Environment Variables:** Upon registration, configure the following environment variables with the Sentinel-2 API credentials:
   
   - `SENTINELSAT_USERNAME`
   - `SENTINELSAT_PASSWORD`

3. **Download Required Data:** Acquire the conservation units and yearly deforestation files from the [TerraBrasilis website](http://terrabrasilis.dpi.inpe.br/en/download-2/).

Once the prerequisites are in place, execute the `step1_download.py` script as follows:


```
python step1_download.py yearly_deforestation.shp conservation_units_amazon_biome.shp <sentinel_output_dataset_path>
```

This command will automatically download the Sentinel-2 tiles, using the provided shapefiles for yearly deforestation and conservation units, and save the output to the specified dataset path.

### Post Step 1: Tile Selection and Subsequent Steps

After completing Step 1, follow the instructions below:

1. **Tile Selection:** Carefully review and manually select the highest quality tiles for use in the subsequent steps of the pipeline. The selected tile IDs are recorded in the `selected.txt` file for exact replication of the data.

2. **Execute Remaining Steps:** With the optimal tiles selected, proceed to execute Steps 2, 3, 4, and 5.

## Training with ChangeFormer

To successfully run the training, the `DataConfig` class in the [`data_config.py`](https://github.com/wgcban/ChangeFormer/blob/main/data_config.py) in [ChangeFormer](https://github.com/wgcban/ChangeFormer) must be updated with the appropriate dataset folder path.

The provided Jupyter Notebook file facilitates the automatic generation of YAML configuration files necessary for executing the training process. An example YAML file is included in the same directory for reference.

## Downloading the Dataset

You can replicate the exact dataset used in this project by utilizing the `selected.txt` file provided in the repository, or download the comprehensive dataset, `DeforestationAmazon.zip`. To ensure a smooth process, make sure you have Git and Git LFS installed on your system. Install Git from [git-scm.com](https://git-scm.com/) and Git LFS from [git-lfs.github.com](https://git-lfs.github.com/).

## Citation

If you use this repository in your research, please cite the following paper:

Alshehri, Mariam, Anes Ouadou, and Grant J. Scott. "Deep Transformer-based Network Deforestation Detection in the Brazilian Amazon Using Sentinel-2 Imagery." IEEE Geoscience and Remote Sensing Letters (2024). [Available online](https://ieeexplore.ieee.org/document/10402113)




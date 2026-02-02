import os
import pandas as pd
import numpy as np
import logging
import sys
from tqdm import tqdm
from pydantic import ValidationError
from src.config import config, ProjectConfig
from src.validation import SpectralValidator, SpectralRecord

os.makedirs(config.log_dir, exist_ok=True) 

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    force=True,
    handlers=[
        logging.StreamHandler(sys.stdout), 
        logging.FileHandler(config.log_file, mode='a', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)    


def parse_folder_metadata(folder_name: str, cfg: ProjectConfig) -> dict:
   
    parts = folder_name.split(cfg.folder_name_delimiter)
    
    metadata = {
        'sample_code': folder_name,
        'botanical': cfg.default_unknown_value,
        'geographic': cfg.default_unknown_value
    }
    
    if len(parts) >= 3:
        metadata['sample_code'] = parts[0]
        metadata['botanical'] = parts[1]
        metadata['geographic'] = parts[2]
        
    return metadata

def create_dataset(cfg: ProjectConfig):
    data_rows = []
    feature_names = None
    primary_key = 1
    
    validator = SpectralValidator(cfg)
    
    logger.info("Starting dataset creation pipeline...")
    logger.info(f"Source Directory: {cfg.raw_data_path}")

    if not os.path.exists(cfg.raw_data_path):
        logger.error(f"CRITICAL: Path not found: {cfg.raw_data_path}")
        raise FileNotFoundError("Missing Data Directory")
    
    all_items = os.listdir(cfg.raw_data_path)
    all_folders = [d for d in all_items if os.path.isdir(os.path.join(cfg.raw_data_path, d))]
    
    valid_folders = []
    for folder in all_folders:
        res = validator.validate_folder_name(folder)
        if res.is_valid:
            valid_folders.append(folder)
        else:
            logger.debug(f"Skipping folder '{folder}': {res.message}")

    if not valid_folders:
        logger.warning("No valid sample folders found based on the configuration pattern.")
        return

    logger.info(f"Found {len(valid_folders)} valid sample folders. Processing...")

    for folder_name in tqdm(valid_folders, desc="Processing Samples", unit="folder"):
        
        class_folder = os.path.join(cfg.raw_data_path, folder_name)
        try:
            files = os.listdir(class_folder)
        except Exception as e:
            logger.warning(f"Could not open folder {folder_name}: {e}")
            continue

        meta = parse_folder_metadata(folder_name, cfg)

        for filename in files:
            
            val_file = validator.validate_filename(filename)
            if not val_file.is_valid:
                continue 

            file_path = os.path.join(class_folder, filename)
            
            try:
                df = pd.read_csv(
                    file_path, 
                    sep=cfg.separator, 
                    skiprows=cfg.skip_lines, 
                    header=None, 
                    engine='python', 
                    encoding='latin-1', 
                    usecols=[0, cfg.data_col_index]
                ) 
                
                df = df.dropna()

                val_struct = validator.validate_structure(df)
                if not val_struct.is_valid:
                    logger.debug(f"Structure Error in {filename}: {val_struct.message}")
                    continue
                                             
                wavelengths = df[0].values
                values = df[cfg.data_col_index].values
                
                if feature_names is None:
                    feature_names = [f"wl_{w:.3f}" for w in wavelengths]
                    logger.info(f"Reference geometry set. Features: {len(feature_names)}")

                val_consist = validator.validate_consistency(len(values))
                if not val_consist.is_valid:
                    logger.debug(f"Skipping {filename}: {val_consist.message}")
                    continue
                
                row_dict = {
                    'id': primary_key,
                    'sample_code': meta['sample_code'],
                    'botanical': meta['botanical'],
                    'geographic': meta['geographic'],
                    'folder_name': folder_name,
                    'filename': filename
                }
                row_dict.update(dict(zip(feature_names, values)))
                
                record = SpectralRecord(**row_dict)
                
                data_rows.append(record.model_dump())
                primary_key += 1
                
            except ValidationError as ve:
                logger.warning(f"Data Validation Failed for {filename}: {ve}")
            except Exception as e:
                logger.warning(f"Failed to read file: {filename}. Cause: {e}")

    if data_rows:
        print("\n") 
        logger.info(f"Creating final dataframe with {len(data_rows)} records...")
        final_df = pd.DataFrame(data_rows)
    
        metadata_cols = ['id', 'sample_code', 'botanical', 'geographic']
        wl_cols = [c for c in final_df.columns if str(c).startswith('wl_')]
        
        final_df = final_df[metadata_cols + wl_cols]
        
        os.makedirs(os.path.dirname(cfg.output_path), exist_ok=True)
        final_df.to_csv(cfg.output_path, index=False)
        
        logger.info("------------------------------------------------")
        logger.info("PROCESS COMPLETED SUCCESSFULLY")
        logger.info(f"Dataset saved at: {cfg.output_path}")
        logger.info(f"Dimensions: {final_df.shape} (Rows, Columns)")
    else:
        logger.warning("The list of data rows is empty. No valid data extracted.")

if __name__ == "__main__":
    create_dataset(config)
import os
from dataclasses import dataclass

@dataclass
class ProjectConfig:
    _src_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir: str = os.path.dirname(_src_dir)
    
    @property
    def raw_data_path(self):
        return os.path.join(self.base_dir, 'data', 'raw')
    
    @property
    def output_path(self):
        return os.path.join(self.base_dir, 'data', 'processed', 'spectral_dataset_enriched.csv')
    
    @property
    def log_dir(self):
        return os.path.join(self.base_dir, 'logs')
    
    @property
    def log_file(self):
        return os.path.join(self.log_dir, 'dataset_creation_pipeline.log')

    skip_lines: int = 8
    data_col_index: int = 1
    separator: str = ';'
    file_extension: str = '.txt'
    
    folder_name_delimiter: str = '_'
    default_unknown_value: str = "Unknown" 

    filename_pattern: str = r"^.+_[a-zA-Z]+_[a-zA-Z]+.*__\d+\.txt$"
    
    foldername_pattern: str = r"^.+_[a-zA-Z]+_[a-zA-Z]+.*"


config = ProjectConfig()
```mermaid
flowchart TD
    %% --- Setup Phase ---
    Start([Start: create_dataset]) --> InitVars[Init: data_rows, feature_names]
    InitVars --> InitValidator[Initialize SpectralValidator]
    InitValidator --> LogStart[Log: Starting creation...]
    
    LogStart --> CheckPath{Raw Data Path Exists?}
    CheckPath -- No --> LogError[Log Critical Error] --> RaiseError[Raise FileNotFoundError] --> Stop([End])
    CheckPath -- Yes --> ScanFolders[Scan 'data/raw' for Subdirectories]

    %% --- Folder Filtering Loop ---
    ScanFolders --> LoopFolders{Loop: All Folders}
    LoopFolders -- Check --> ValFolder[validator.validate_folder_name]
    ValFolder -- Valid --> AddValid[Add to 'valid_folders'] --> LoopFolders
    ValFolder -- Invalid --> LogSkipFolder[Log Debug: Skip Folder] --> LoopFolders
    
    LoopFolders -- Done --> CheckValid{Any Valid Folders?}
    CheckValid -- No --> LogWarnNoFolders[Log Warning: No valid folders] --> Stop
    CheckValid -- Yes --> LogInfoCount[Log Info: Found N folders] --> MainLoop
    
    %% --- Main Processing Loop ---
    subgraph Processing ["Main Processing Loop (tqdm)"]
        MainLoop[For each Valid Folder] --> ParseMeta[parse_folder_metadata]
        ParseMeta --> ListFiles[os.listdir]
        
        ListFiles --> InnerLoop{For each File}
        InnerLoop --> ValFile[validator.validate_filename]
        ValFile -- Invalid --> InnerLoop
        ValFile -- Valid --> ReadCSV["pd.read_csv (Latin-1)"]
        
        ReadCSV -- Error --> CatchEx[Catch Exception] --> LogWarnFile[Log Warning] --> InnerLoop
        ReadCSV -- Success --> DropNA["df.dropna()"]
        
        DropNA --> ValStruct[validator.validate_structure]
        ValStruct -- Invalid --> LogWarnStruct[Log Warning: Structure Error] --> InnerLoop
        ValStruct -- Valid --> ExtractVals[Extract Wavelengths & Values]
        
        ExtractVals --> CheckRef{First File?}
        CheckRef -- Yes --> SetRef[Set Reference Feature Names] --> ValConsist
        CheckRef -- No --> ValConsist[validator.validate_consistency]
        
        ValConsist -- Invalid --> LogDebugDim[Log Debug: Mismatch] --> InnerLoop
        ValConsist -- Valid --> BuildRow[Create Dictionary Row]
        
        BuildRow --> Pydantic[SpectralRecord Validation]
        Pydantic -- Error --> LogWarnPydantic[Log Validation Error] --> InnerLoop
        Pydantic -- Success --> AppendRow[Append to data_rows] --> IncID[Increment ID] --> InnerLoop
    end
    
    InnerLoop -- Folder Done --> MainLoop
    
    %% --- Export Phase ---
    MainLoop -- All Done --> CheckData{data_rows not empty?}
    CheckData -- No --> LogWarnEmpty[Log Warning: No data rows] --> Stop
    CheckData -- Yes --> CreateDF[Create Pandas DataFrame]
    
    CreateDF --> SortCols["Reorder Columns (Meta + Wavelengths)"]
    SortCols --> MakeDir[os.makedirs processed]
    MakeDir --> SaveCSV[df.to_csv]
    SaveCSV --> LogSuccess[Log Success] --> Stop
    
    %% Colors
    style Start fill:#90EE90,stroke:#333
    style Stop fill:#FF7F7F,stroke:#333
    style Processing fill:stroke:#333,stroke-dasharray: 5 5
    style RaiseError fill:#FFB6C1
    style LogSuccess fill:#90EE90
```
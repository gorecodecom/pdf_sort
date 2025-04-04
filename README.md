# Document Sorter

A smart document organization tool that automatically sorts your documents into categorized folders based on filenames. The program learns from your choices and maintains consistent folder structures across different locations.

## Features

### Smart Document Organization
- Automatically sorts documents (PDF, JPG, PNG, JPEG) into categorized folders
- Extracts document types from filenames
- Creates year-based subfolders automatically (e.g., 2024, 2023) based on dates in filenames
- Maintains consistent folder numbering (e.g., 01_Vertrag, 02_Information)

### Learning System
- Learns from your categorization choices
- Remembers document types for each category
- Stores learning data in a `.pdf_sorter_knowledge.json` file
- Option to store learning data locally or in cloud storage

### Cloud Storage Support
- Works with major cloud storage providers:
  - iCloud Drive
  - Google Drive
  - Dropbox
  - OneDrive
- Handles cloud sync states and ensures files are fully downloaded
- Maintains consistent folder structure across different locations

### Smart Naming
- Detects personal names in filenames
- Handles various date formats
- Maintains consistent folder naming conventions
- Automatically numbers folders (01_, 02_, etc.)

### Folder Structure Synchronization
- Automatically creates category folders from learning data
- Renames existing folders to match learned structure
- Creates year subfolders as needed
- Preserves existing files during folder reorganization

## Requirements

```
python >= 3.6
PyPDF2
scikit-learn
numpy
```

## Installation

1. Clone or download this repository
2. Install the required packages:
```bash
pip install -r requirements.txt
```

## Usage

1. Run the program:
```bash
python pdf_sorter.py
```

2. Enter the path to your documents folder when prompted
   - For cloud storage, use the full path
   - For iCloud: `/Users/username/Library/Mobile Documents/com~apple~CloudDocs/your/folder`
   - For local folders: `/path/to/your/documents`

3. The program will:
   - Create or sync folder structure
   - Process each document
   - Ask for categorization when unsure
   - Create year subfolders as needed
   - Move files to appropriate locations

4. After processing one folder, you can choose to:
   - Process another folder
   - Exit the program

## Folder Structure Example

```
Documents/
├── 01_Vertrag/
│   ├── 2024/
│   │   └── 20240316_contract.pdf
│   └── 2023/
│       └── 20230515_agreement.pdf
├── 02_Information/
│   └── 2024/
│       └── 20240112_info.pdf
└── 03_Rechnung/
    ├── 2024/
    │   └── 20240201_invoice.pdf
    └── other_invoice.pdf
```

## Learning Data

- The program stores learning data in `.pdf_sorter_knowledge.json`
- You can choose to store this file:
  - In your cloud storage (for sync across devices)
  - In your local user directory
  - In a different cloud storage location

## Tips

1. **Cloud Storage**
   - Ensure folders are synced locally before processing
   - The program will wait for files to finish downloading
   - Use "Copy as Pathname" from Finder for accurate paths

2. **Naming Conventions**
   - Include dates in YYYYMMDD format for automatic year sorting
   - Use descriptive names for better categorization
   - Separate words with underscores or spaces

3. **Categories**
   - Categories are automatically numbered (01_, 02_, etc.)
   - Existing folders will be renamed to match the structure
   - Empty folders will be created for known categories

## Error Handling

The program handles various scenarios:
- Cloud sync issues
- Permission problems
- Invalid paths
- File access errors
- Folder creation/renaming issues

## Notes

- The program preserves existing files during folder reorganization
- Year subfolders are only created when needed
- Learning data can be shared across different folders
- The program maintains consistent structure across all locations 
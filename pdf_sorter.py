#!/usr/bin/env python3
import os
import shutil
import json
from pathlib import Path
from typing import Dict, List, Set, Tuple
import PyPDF2
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from datetime import datetime
import re
import time

class PDFSorter:
    # Cloud storage path indicators and default paths
    CLOUD_PATHS = {
        'icloud': {
            'indicators': ['Library/Mobile Documents/com~apple~CloudDocs'],
            'default_path': '~/Library/Mobile Documents/com~apple~CloudDocs'
        },
        'google_drive': {
            'indicators': ['Google Drive', 'GoogleDrive'],
            'default_path': '~/Google Drive'
        },
        'dropbox': {
            'indicators': ['Dropbox'],
            'default_path': '~/Dropbox'
        },
        'onedrive': {
            'indicators': ['OneDrive'],
            'default_path': '~/OneDrive'
        }
    }

    def __init__(self, source_dir: str):
        """
        Initialize the PDF sorter with a source directory.
        
        Args:
            source_dir (str): Path to the directory containing PDF files
        """
        self.source_dir = Path(source_dir)
        self.cloud_type = self._check_if_cloud_path(source_dir)
        self.knowledge_file = self._get_knowledge_file_path()
        self.learned_categories = self._load_learned_categories()
        self.vectorizer = TfidfVectorizer(
            min_df=1, 
            stop_words=['the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'with'],
            token_pattern=r'(?u)\b\w+\b'
        )
        self.is_cloud_path = self.cloud_type is not None
        
        # Sync folder structure with knowledge file
        self._sync_folder_structure()
        
    def _get_knowledge_file_path(self) -> Path:
        """Determine the path for the knowledge file."""
        # First, check if knowledge file exists in the current cloud directory
        if self.cloud_type:
            cloud_base = self._get_cloud_base_path(self.cloud_type)
            if cloud_base:
                cloud_knowledge_file = Path(cloud_base) / ".pdf_sorter_knowledge.json"
                if cloud_knowledge_file.exists():
                    return cloud_knowledge_file

        # Second, check if knowledge file exists in local user directory
        local_knowledge_file = Path.home() / ".pdf_sorter_knowledge.json"
        if local_knowledge_file.exists():
            # If found in local directory, ask if user wants to move it to cloud
            if self.cloud_type:
                print("\nFound existing learning data in local directory.")
                move = input("Would you like to move it to your cloud storage? (y/n): ").lower()
                if move == 'y':
                    cloud_base = self._get_cloud_base_path(self.cloud_type)
                    if cloud_base:
                        cloud_knowledge_file = Path(cloud_base) / ".pdf_sorter_knowledge.json"
                        try:
                            # Ensure cloud directory exists
                            cloud_knowledge_file.parent.mkdir(parents=True, exist_ok=True)
                            # Copy file to cloud and remove local
                            shutil.copy2(local_knowledge_file, cloud_knowledge_file)
                            os.remove(local_knowledge_file)
                            print("Learning data moved to cloud storage.")
                            return cloud_knowledge_file
                        except Exception as e:
                            print(f"Error moving file to cloud: {str(e)}")
                            print("Continuing with local file.")
                            return local_knowledge_file
            return local_knowledge_file

        # If no existing file found, ask user where to store it
        print("\nWhere would you like to store the learning data?")
        if self.cloud_type:
            print(f"1. Current cloud storage ({self.cloud_type.title()})")
        print("2. Local user directory")
        print("3. Different cloud storage")

        while True:
            choice = input("Enter number: ").strip()
            
            # Use current cloud storage
            if choice == "1" and self.cloud_type:
                cloud_base = self._get_cloud_base_path(self.cloud_type)
                if cloud_base:
                    return Path(cloud_base) / ".pdf_sorter_knowledge.json"
                print("Error accessing current cloud storage.")
                continue
                
            # Use local directory
            elif choice == "2":
                return Path.home() / ".pdf_sorter_knowledge.json"
                
            # Choose different cloud storage
            elif choice == "3":
                print("\nChoose cloud storage:")
                print("1. iCloud Drive")
                print("2. Google Drive")
                print("3. Dropbox")
                print("4. OneDrive")
                
                cloud_choice = input("Enter number (1-4): ").strip()
                cloud_type = {
                    "1": "icloud",
                    "2": "google_drive",
                    "3": "dropbox",
                    "4": "onedrive"
                }.get(cloud_choice)
                
                if cloud_type:
                    cloud_base = self._get_cloud_base_path(cloud_type)
                    if cloud_base:
                        return Path(cloud_base) / ".pdf_sorter_knowledge.json"
                print("Selected cloud storage not found or not accessible.")
                continue
                
            print("Invalid choice. Please try again.")
            
    def _get_cloud_base_path(self, cloud_type: str) -> str:
        """Get the base path for a cloud service."""
        if cloud_type not in self.CLOUD_PATHS:
            return None
            
        # Try the default path first
        default_path = os.path.expanduser(self.CLOUD_PATHS[cloud_type]['default_path'])
        if os.path.exists(default_path):
            return default_path
            
        # For iCloud, try the full path
        if cloud_type == 'icloud':
            icloud_path = os.path.expanduser("~/Library/Mobile Documents/com~apple~CloudDocs")
            if os.path.exists(icloud_path):
                return icloud_path
                
        return None
        
    def _check_if_cloud_path(self, path: str) -> str:
        """
        Check if the path is in a cloud storage directory.
        Returns the cloud service name or None.
        """
        path = path.lower()
        for service, data in self.CLOUD_PATHS.items():
            if any(indicator.lower() in path for indicator in data['indicators']):
                return service
        return None
        
    def _load_learned_categories(self) -> Dict:
        """Load previously learned categories and their document types."""
        if self.knowledge_file.exists():
            try:
                with open(self.knowledge_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except json.JSONDecodeError:
                return {}
        return {}
        
    def _save_learned_categories(self):
        """Save learned categories to file."""
        with open(self.knowledge_file, 'w', encoding='utf-8') as f:
            json.dump(self.learned_categories, f, ensure_ascii=False, indent=2)
            
    def _extract_document_type(self, filename: str) -> str:
        """Extract the document type from filename (after date if present)."""
        # Remove file extension
        name = Path(filename).stem
        
        # Split by common separators
        parts = re.split(r'[_\s-]', name)
        
        # Remove date-like parts (assuming YYYYMMDD or DDMMYYYY format)
        parts = [p for p in parts if not re.match(r'^\d{8}$', p)]
        
        # Join remaining parts
        return ' '.join(parts)
            
    def _suggest_category(self, doc_type: str) -> str:
        """Suggest a category based on learned patterns."""
        if not self.learned_categories:
            return None
            
        # Check each category's known document types
        for category, data in self.learned_categories.items():
            if doc_type.lower() in [t.lower() for t in data['document_types']]:
                return category
                
        return None
        
    def _ask_for_category(self, filename: str, doc_type: str) -> str:
        """Ask user for the correct category and learn from it."""
        print(f"\nNeed help categorizing: {filename}")
        print(f"Document type appears to be: {doc_type}")
        
        # Check if there might be a personal name
        words = doc_type.split()
        if len(words) > 1:
            confirm = input("This filename contains multiple words. Is any part a personal name? (y/n): ").lower()
            if confirm == 'y':
                print("\nPlease select the actual document type (excluding personal names):")
                for i, word in enumerate(words, 1):
                    print(f"{i}. {word}")
                while True:
                    try:
                        choice = int(input("Enter number: "))
                        if 1 <= choice <= len(words):
                            doc_type = words[choice-1]
                            break
                    except ValueError:
                        pass
                    print("Please enter a valid number.")
        
        # Show existing categories
        if self.learned_categories:
            print("\nExisting categories:")
            categories = list(self.learned_categories.keys())
            for i, category in enumerate(categories, 1):
                print(f"{i}. {category}")
                # Show known document types for this category
                doc_types = self.learned_categories[category]['document_types']
                if doc_types:
                    print(f"   Known types: {', '.join(doc_types)}")
            print("n. Create new category")
            
            choice = input("\nChoose category number or 'n' for new: ").strip().lower()
            
            if choice == 'n':
                return self._create_new_category(doc_type)
            try:
                idx = int(choice) - 1
                category = categories[idx]
                self._update_category_knowledge(category, doc_type)
                return category
            except (ValueError, IndexError):
                print("Invalid choice. Creating new category.")
                return self._create_new_category(doc_type)
        else:
            return self._create_new_category(doc_type)
            
    def _create_new_category(self, doc_type: str) -> str:
        """Create a new category and learn from it."""
        while True:
            category = input("\nEnter new category name (without number prefix): ").strip()
            if category and not any(c in category for c in '\\/:*?"<>|'):
                break
            print("Invalid category name. Avoid special characters.")
            
        # Get current categories and determine next number
        existing_categories = self._get_ordered_categories()
        next_num = len(existing_categories) + 1
        
        # Create new category with number prefix
        numbered_category = f"{str(next_num).zfill(2)}_{category}"
        
        self.learned_categories[numbered_category] = {
            'document_types': [doc_type],
            'created_at': datetime.now().isoformat()
        }
        self._save_learned_categories()
        
        # Create the folder
        try:
            (self.source_dir / numbered_category).mkdir(exist_ok=True)
            print(f"Created folder: {numbered_category}")
        except Exception as e:
            print(f"Error creating folder {numbered_category}: {str(e)}")
            
        return numbered_category
        
    def _update_category_knowledge(self, category: str, doc_type: str):
        """Update category knowledge with new document type."""
        if category not in self.learned_categories:
            self.learned_categories[category] = {
                'document_types': [],
                'created_at': datetime.now().isoformat()
            }
            
        # Add new document type if not already known
        if doc_type.lower() not in [t.lower() for t in self.learned_categories[category]['document_types']]:
            self.learned_categories[category]['document_types'].append(doc_type)
            self._save_learned_categories()
        
    def _wait_for_file_sync(self, file_path: Path, timeout: int = 30) -> bool:
        """
        Wait for a file to be fully synced (no size changes for a period).
        Returns True if file appears stable, False if timeout reached.
        """
        start_time = time.time()
        last_size = -1
        stable_count = 0
        
        while time.time() - start_time < timeout:
            try:
                current_size = file_path.stat().st_size
                if current_size == last_size:
                    stable_count += 1
                    if stable_count >= 3:  # File size stable for 3 checks
                        return True
                else:
                    stable_count = 0
                last_size = current_size
                time.sleep(1)
            except FileNotFoundError:
                time.sleep(1)
                continue
            
        return False

    def _is_cloud_file_ready(self, file_path: Path) -> bool:
        """Check if a cloud-stored file is ready for processing."""
        if not self.is_cloud_path:
            return True
            
        try:
            # Check if file exists and is readable
            if not file_path.exists():
                print(f"\nWaiting for cloud to sync {file_path.name}...")
                return False
                
            # Check if file is fully downloaded (not just a placeholder)
            if file_path.stat().st_size == 0:
                print(f"\nWaiting for {file_path.name} to download...")
                return False
                
            # Wait for file to stabilize (finish syncing)
            if not self._wait_for_file_sync(file_path):
                print(f"\nWarning: {file_path.name} may not be fully synced")
                confirm = input("Process anyway? (y/n): ").lower()
                return confirm == 'y'
                
            return True
            
        except (PermissionError, FileNotFoundError) as e:
            print(f"\nError accessing {file_path.name}: {str(e)}")
            return False

    def _extract_year_from_filename(self, filename: str) -> str:
        """Extract year from filename if present."""
        # Look for 4-digit year patterns
        year_match = re.search(r'20\d{2}', filename)
        if year_match:
            return year_match.group(0)
        return None

    def _ensure_year_subfolder(self, category_dir: Path, year: str) -> Path:
        """Create year subfolder if it doesn't exist and return the path."""
        year_dir = category_dir / year
        try:
            year_dir.mkdir(exist_ok=True)
            return year_dir
        except Exception as e:
            print(f"\nError creating year folder {year}: {str(e)}")
            return category_dir  # Fallback to category directory if creation fails

    def sort_pdfs(self) -> Dict[str, List[str]]:
        """Sort files into appropriate categories and year subfolders."""
        results = defaultdict(lambda: defaultdict(list))
        
        # Get list of files
        try:
            files = []
            # Recursively find all files in source directory and its subdirectories
            for root, _, filenames in os.walk(self.source_dir):
                root_path = Path(root)
                # Skip if the current directory is already a year subfolder
                if root_path.name.isdigit() and len(root_path.name) == 4:
                    continue
                for filename in filenames:
                    if filename.lower().endswith(('.pdf', '.jpg', '.png', '.jpeg')):
                        files.append(root_path / filename)
        except PermissionError:
            print(f"\nError: Cannot access {self.source_dir}")
            print("Please check if you have permission to access this cloud folder")
            return dict(results)
            
        total_files = len(files)
        
        if not total_files:
            print("No files found in the directory.")
            return dict(results)
            
        if self.is_cloud_path:
            print(f"\nWorking with {self.cloud_type.title()} folder. Files may take longer to process.")
            
        # Process each file
        for i, file in enumerate(files, 1):
            print(f"\rProcessing file {i}/{total_files}: {file.name}", end="")
            
            # For cloud files, ensure file is ready
            if not self._is_cloud_file_ready(file):
                print(f"\nSkipping {file.name} - not ready for processing")
                continue
                
            # Extract document type from filename
            doc_type = self._extract_document_type(file.name)
            
            # Try to determine category
            category = self._suggest_category(doc_type)
            
            # If unsure, ask user
            if category is None:
                print()  # New line for user interaction
                category = self._ask_for_category(file.name, doc_type)
                
            # Create category folder if needed
            category_dir = self.source_dir / category
            try:
                category_dir.mkdir(exist_ok=True)
            except PermissionError:
                print(f"\nError: Cannot create folder {category} in cloud directory")
                print("Please check your permissions or create the folder manually")
                continue

            # Extract year from filename and determine target directory
            year = self._extract_year_from_filename(file.name)
            if year:
                target_dir = self._ensure_year_subfolder(category_dir, year)
            else:
                target_dir = category_dir

            # Move file
            target_path = target_dir / file.name
            try:
                # For cloud storage, ensure we have write permission
                if self.is_cloud_path:
                    print(f"\nMoving {file.name} to {target_dir.relative_to(self.source_dir)}...")
                shutil.move(str(file), str(target_path))
                results[category][year if year else "no_year"].append(file.name)
            except Exception as e:
                print(f"\nError moving file {file}: {str(e)}")
                if self.is_cloud_path:
                    print("This might be due to cloud sync issues. Please try again in a few moments.")
                
        print()  # New line after progress indicator
        
        # Format results for printing
        formatted_results = {}
        for category, year_files in results.items():
            formatted_results[category] = []
            for year, files in year_files.items():
                if year == "no_year":
                    formatted_results[category].extend(files)
                else:
                    formatted_results[category].extend([f"{year}/{file}" for file in files])
        
        return formatted_results

    def _get_ordered_categories(self) -> List[str]:
        """Get categories in order, prefixed with numbers if not already prefixed."""
        categories = list(self.learned_categories.keys())
        
        # Check if categories are already numbered
        numbered = all(re.match(r'^\d+_', cat) for cat in categories)
        
        if not numbered:
            # Sort alphabetically first
            categories.sort()
            # Add numbers as prefix
            categories = [f"{str(i+1).zfill(2)}_{cat}" for i, cat in enumerate(categories)]
            
            # Update learned categories with new names
            new_learned = {}
            for i, old_cat in enumerate(self.learned_categories.keys()):
                new_cat = categories[i]
                new_learned[new_cat] = self.learned_categories[old_cat]
            self.learned_categories = new_learned
            self._save_learned_categories()
            
        return sorted(categories)

    def _sync_folder_structure(self):
        """Synchronize folder structure with the knowledge file."""
        if not self.learned_categories:
            return

        try:
            # Get ordered categories
            ordered_categories = self._get_ordered_categories()
            
            # Get existing folders
            existing_folders = [f.name for f in self.source_dir.iterdir() if f.is_dir()]
            
            # Create mapping of folder types (without numbers) to full names
            existing_folder_map = {
                re.sub(r'^\d+_', '', folder.lower()): folder
                for folder in existing_folders
            }
            
            # Process each category from knowledge file
            for category in ordered_categories:
                category_path = self.source_dir / category
                base_name = re.sub(r'^\d+_', '', category.lower())
                
                # Check if a similar folder exists (ignoring number prefix)
                if base_name in existing_folder_map:
                    existing_folder = existing_folder_map[base_name]
                    if existing_folder != category:
                        # Rename folder if it doesn't match the exact name
                        old_path = self.source_dir / existing_folder
                        try:
                            old_path.rename(category_path)
                            print(f"Renamed folder: {existing_folder} → {category}")
                        except Exception as e:
                            print(f"Error renaming folder {existing_folder}: {str(e)}")
                else:
                    # Create new folder if it doesn't exist
                    try:
                        category_path.mkdir(exist_ok=True)
                        print(f"Created folder: {category}")
                    except Exception as e:
                        print(f"Error creating folder {category}: {str(e)}")

        except Exception as e:
            print(f"Error synchronizing folder structure: {str(e)}")
            print("Please check folder permissions and try again.")

def get_valid_directory() -> str:
    """Ask user for a directory path and validate it."""
    while True:
        path = input("\nEnter the path to your documents folder: ").strip()
        
        # Handle paths with quotes
        path = path.strip('"\'')
        
        # Convert ~ to full home directory path if present
        if path.startswith("~"):
            path = os.path.expanduser(path)
            
        # Convert relative path to absolute path
        path = os.path.abspath(path)
        
        # Special handling for iCloud paths
        if "Mobile Documents/com~apple~CloudDocs" in path:
            # Ensure the path is properly formatted for iCloud
            icloud_base = os.path.expanduser("~/Library/Mobile Documents/com~apple~CloudDocs")
            if not path.startswith(icloud_base):
                # Try to fix the path by replacing the iCloud portion
                path = path.replace("Library/Mobile Documents/com~apple~CloudDocs", 
                                  icloud_base.replace(os.path.expanduser("~"), ""))
                path = os.path.expanduser(path)
        
        try:
            # Try to resolve any symlinks and normalize the path
            path = str(Path(path).resolve())
            
            if not os.path.exists(path):
                print(f"Error: The path '{path}' does not exist.")
                print("If this is an iCloud folder, make sure it's synced locally.")
                print("Try copying the path directly from Finder (right-click folder → Option key → Copy as Pathname)")
                continue
                
            if not os.path.isdir(path):
                print(f"Error: '{path}' is not a directory.")
                continue
                
            # Try to list directory contents to verify permissions
            try:
                files = [f for f in os.listdir(path) 
                        if f.lower().endswith(('.pdf', '.jpg', '.png', '.jpeg'))]
                if not files:
                    print(f"Warning: No supported files found in '{path}'")
                    confirm = input("Do you want to continue anyway? (y/n): ").lower()
                    if confirm != 'y':
                        continue
            except PermissionError:
                print(f"Error: Cannot access '{path}'")
                print("Please check your permissions for this folder")
                continue
                
            return path
            
        except Exception as e:
            print(f"Error with path '{path}': {str(e)}")
            print("Please make sure the path is correct and accessible")
            continue

def main():
    print("Document Sorter")
    print("==============")
    
    while True:
        # Get and validate the source directory from user input
        source_dir = get_valid_directory()
        
        # Create and run the sorter
        sorter = PDFSorter(source_dir)
        results = sorter.sort_pdfs()
        
        # Print results
        print("\nSorting Results:")
        print("--------------")
        if not any(results.values()):
            print("No files were moved.")
        else:
            for category, files in results.items():
                if files:
                    print(f"\n{category}:")
                    for file in files:
                        print(f"  - {file}")
        
        # Ask if user wants to process another folder
        print("\nWould you like to sort another folder?")
        choice = input("Enter 'y' for yes, any other key to exit: ").lower().strip()
        if choice != 'y':
            print("\nThank you for using Document Sorter!")
            break

if __name__ == "__main__":
    main() 
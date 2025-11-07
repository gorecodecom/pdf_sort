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

    # Default categories that should always be available
    DEFAULT_CATEGORIES = {
        "01 Antrag": {
            'document_types': ['antrag', 'application', 'bewerbung', 'formular', 'form'],
            'created_at': '2024-01-01T00:00:00'
        },
        "02 Bescheid": {
            'document_types': ['bescheid', 'decision', 'entscheidung', 'beschluss', 'notice'],
            'created_at': '2024-01-01T00:00:00'
        },
        "03 Vertrag": {
            'document_types': ['vertrag', 'contract', 'vereinbarung', 'agreement'],
            'created_at': '2024-01-01T00:00:00'
        },
        "04 Rechnung": {
            'document_types': ['rechnung', 'invoice', 'bill', 'faktura', 'beleg', 'quittung'],
            'created_at': '2024-01-01T00:00:00'
        },
        "05 Information": {
            'document_types': ['information', 'info', 'infoblatt', 'mitteilung', 'benachrichtigung'],
            'created_at': '2024-01-01T00:00:00'
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

        # Ensure default categories are always available
        self._ensure_default_categories()

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
        path = path.lower().replace('\\', '/')  # Normalize path separators
        for service, data in self.CLOUD_PATHS.items():
            if any(indicator.lower() in path for indicator in data['indicators']):
                return service
        # Check for common Windows cloud paths
        if 'icloud' in path or 'icloudrive' in path:
            return 'icloud'
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

    def _ensure_default_categories(self):
        """Ensure default categories are present in learned categories."""
        # Migration mapping for old categories to new ones
        migration_map = {
            "01 Vertrag": "03 Vertrag",
            "02 Information": "05 Information",
            "03 Rechnung": "04 Rechnung"
        }

        # Migrate old categories to new structure
        for old_cat, new_cat in migration_map.items():
            if old_cat in self.learned_categories and new_cat not in self.learned_categories:
                print(f"Migrating category: {old_cat} → {new_cat}")
                self.learned_categories[new_cat] = self.learned_categories[old_cat]
                del self.learned_categories[old_cat]

        # Add default categories if they don't exist
        for category, data in self.DEFAULT_CATEGORIES.items():
            if category not in self.learned_categories:
                # Check if there's an underscore version that should be updated
                underscore_version = category.replace(' ', '_')
                if underscore_version in self.learned_categories:
                    # Move data from underscore version to space version
                    self.learned_categories[category] = self.learned_categories[underscore_version]
                    del self.learned_categories[underscore_version]
                else:
                    # Add new default category
                    self.learned_categories[category] = data.copy()

        # Save updated categories
        self._save_learned_categories()

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

        doc_type_lower = doc_type.lower()

        # Check each category's known document types
        for category, data in self.learned_categories.items():
            for known_type in data['document_types']:
                # Check for exact match or partial match
                if (known_type.lower() in doc_type_lower or
                    doc_type_lower in known_type.lower()):
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

        # Show predefined categories only
        print("\nAvailable categories:")
        categories = ["01 Antrag", "02 Bescheid", "03 Vertrag", "04 Rechnung", "05 Information"]
        for i, category in enumerate(categories, 1):
            print(f"{i}. {category}")
            # Show known document types for this category
            if category in self.learned_categories:
                doc_types = self.learned_categories[category]['document_types']
                if doc_types:
                    print(f"   Known types: {', '.join(doc_types)}")

        while True:
            try:
                choice = int(input("\nChoose category number (1-5): "))
                if 1 <= choice <= 5:
                    category = categories[choice - 1]
                    self._update_category_knowledge(category, doc_type)
                    return category
                else:
                    print("Please enter a number between 1 and 5.")
            except ValueError:
                print("Please enter a valid number.")


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

    def _format_filename(self, filename: str) -> str:
        """
        Format filename to change date format from yyyymmdd to yyyy-mm-dd.

        Args:
            filename (str): Original filename

        Returns:
            str: Formatted filename with yyyy-mm-dd date format
        """
        # Get file extension
        path_obj = Path(filename)
        stem = path_obj.stem
        extension = path_obj.suffix

        # Look for date pattern yyyymmdd at the beginning of filename
        date_pattern = r'^(\d{4})(\d{2})(\d{2})(_|-)(.+)$'
        match = re.match(date_pattern, stem)

        if match:
            year, month, day, separator, rest = match.groups()
            # Format as yyyy-mm-dd rest (replace separator with space)
            formatted_stem = f"{year}-{month}-{day} {rest}"
            return formatted_stem + extension

        # If no date pattern found, return original filename
        return filename

    def _rename_file_if_needed(self, file_path: Path, target_dir: Path) -> Path:
        """
        Rename file with proper date formatting and return the new target path.

        Args:
            file_path (Path): Current file path
            target_dir (Path): Directory where file will be moved

        Returns:
            Path: Final target path with formatted filename
        """
        original_filename = file_path.name
        formatted_filename = self._format_filename(original_filename)

        # Check if filename was changed
        if formatted_filename != original_filename:
            print(f"\n  Renaming: {original_filename} → {formatted_filename}")

        return target_dir / formatted_filename

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

            # Rename file if needed and get target path
            target_path = self._rename_file_if_needed(file, target_dir)

            # Debug output
            print(f"\n  DEBUG:")
            print(f"  Source file: {file}")
            print(f"  Source exists: {file.exists()}")
            print(f"  Target dir: {target_dir}")
            print(f"  Target path: {target_path}")
            print(f"  Target dir exists: {target_dir.exists()}")

            # Check if file already exists at target location
            if target_path.exists():
                print(f"\n  File already exists: {target_path.name}")
                choice = input("  Overwrite? (y/n): ").lower()
                if choice != 'y':
                    print(f"  Skipped: {file.name}")
                    continue

            try:
                # Ensure target directory exists
                print(f"  Creating target directory: {target_dir}")
                target_dir.mkdir(parents=True, exist_ok=True)
                print(f"  Target directory created/exists: {target_dir.exists()}")

                # For cloud storage, show additional info
                if self.is_cloud_path:
                    print(f"\n  Moving {file.name} to {target_dir.relative_to(self.source_dir)}...")

                print(f"  Attempting to move file...")
                print(f"  From: {file}")
                print(f"  To: {target_path}")

                # Perform the move operation
                shutil.move(str(file), str(target_path))

                print(f"  Move command executed")

                # For iCloud, add delay to allow sync
                if self.is_cloud_path:
                    print(f"  Waiting for iCloud sync...")
                    time.sleep(2)

                    # Force file system refresh by trying to list directory
                    try:
                        list(self.source_dir.iterdir())
                        list(target_dir.iterdir())
                    except:
                        pass

                # Verify the move was successful (check multiple times for iCloud)
                max_checks = 5 if self.is_cloud_path else 1
                for check_num in range(max_checks):
                    if check_num > 0:
                        time.sleep(1)
                        print(f"  Verification attempt {check_num + 1}...")

                    source_exists_after = file.exists()
                    target_exists_after = target_path.exists()

                    print(f"  After move - Source exists: {source_exists_after}")
                    print(f"  After move - Target exists: {target_exists_after}")

                    if target_exists_after and not source_exists_after:
                        results[category][year if year else "no_year"].append(target_path.name)
                        print(f"  ✓ Successfully moved to: {target_path.relative_to(self.source_dir)}")
                        break
                    elif check_num == max_checks - 1:
                        print(f"  ✗ Move operation failed or incomplete after {max_checks} attempts")
                        if self.is_cloud_path:
                            print(f"  This might be an iCloud sync delay. Please check the folders manually.")
                        break

            except Exception as e:
                print(f"\n  ERROR moving file {file}: {str(e)}")
                print(f"  Error type: {type(e).__name__}")
                import traceback
                print(f"  Traceback: {traceback.format_exc()}")
                if self.is_cloud_path:
                    print("  This might be due to cloud sync issues. Please try again in a few moments.")
                else:
                    print("  Please check file permissions and ensure the target directory is writable.")

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


    def _sync_folder_structure(self):
        """Synchronize folder structure with the knowledge file and migrate all old variants to the new structure."""
        if not self.learned_categories:
            return

        try:
            # Zielstruktur: Mapping von Kategorie-Basisnamen auf Zielordnernamen
            canonical_folders = {
                "antrag": "01 Antrag",
                "bescheid": "02 Bescheid",
                "vertrag": "03 Vertrag",
                "rechnung": "04 Rechnung",
                "information": "05 Information"
            }

            # Alle existierenden Ordner im Quellverzeichnis erfassen
            existing_folders = [f for f in self.source_dir.iterdir() if f.is_dir()]

            # Mapping: Basisname (ohne Nummer, Unterstrich, etc.) → Liste[Ordner]
            folder_variants = {}
            for folder in existing_folders:
                # Basisname extrahieren (z.B. "01_Vertrag" → "vertrag", "Vertrag" → "vertrag")
                name = folder.name.lower().replace('_', ' ').strip()
                # Entferne führende Nummern und Leerzeichen
                name = re.sub(r'^[0-9]+[\s_]*', '', name)
                # Nur den ersten Begriff nehmen (z.B. "vertrag" aus "vertrag alt")
                base = name.split()[0] if name.split() else name
                if base in canonical_folders:
                    folder_variants.setdefault(base, []).append(folder)

            # Für jede Kategorie: alle Varianten in Zielordner zusammenführen
            for base, target_name in canonical_folders.items():
                target_path = self.source_dir / target_name
                # Zielordner ggf. anlegen
                if not target_path.exists():
                    try:
                        target_path.mkdir(exist_ok=True)
                        print(f"Created folder: {target_name}")
                    except Exception as e:
                        print(f"Error creating folder {target_name}: {str(e)}")
                        continue
                # Alle Varianten (außer Zielordner selbst) migrieren
                for folder in folder_variants.get(base, []):
                    if folder == target_path:
                        continue
                    # Dateien/Unterordner verschieben
                    for item in folder.iterdir():
                        dest = target_path / item.name
                        # Falls Datei schon existiert, umbenennen
                        if dest.exists():
                            # Füge Suffix hinzu, um Kollision zu vermeiden
                            stem, ext = os.path.splitext(item.name)
                            for i in range(1, 100):
                                new_name = f"{stem}_alt{i}{ext}"
                                new_dest = target_path / new_name
                                if not new_dest.exists():
                                    dest = new_dest
                                    break
                        try:
                            if item.is_dir():
                                shutil.move(str(item), str(dest))
                            else:
                                shutil.move(str(item), str(dest))
                        except Exception as e:
                            print(f"Error moving {item} to {dest}: {str(e)}")
                    # Alten Ordner löschen
                    try:
                        folder.rmdir()
                        print(f"Removed old folder: {folder.name}")
                    except Exception as e:
                        print(f"Error removing old folder {folder.name}: {str(e)}")

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

            # For iCloud, add verification reminder
            if sorter.is_cloud_path:
                print(f"\n📝 Note: If you're using iCloud, please verify that files have been moved.")
                print(f"   iCloud sync can sometimes delay file system operations.")
                print(f"   You can check the folders manually to confirm the moves completed.")

        # Ask if user wants to process another folder
        print("\nWould you like to sort another folder?")
        choice = input("Enter 'y' for yes, any other key to exit: ").lower().strip()
        if choice != 'y':
            print("\nThank you for using Document Sorter!")
            break

if __name__ == "__main__":
    main()

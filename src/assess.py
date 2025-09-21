#!/usr/bin/env python3
"""
Phase 1: Assessment Script
Performs a read-only scan of the source directory and generates a report.
"""

import os
import csv
import mimetypes
from pathlib import Path
from typing import List, Dict, Any
import argparse
from datetime import datetime

import sys
sys.path.append(str(Path(__file__).parent.parent))
from config.settings import DEFAULT_SOURCE_DIR, SUPPORTED_EXTENSIONS

# Use environment variable for dynamic source directory
SOURCE_DIR = Path(os.getenv('RAG_SOURCE_DIR', DEFAULT_SOURCE_DIR))
ASSESSMENT_REPORT = Path(os.getenv('RAG_ASSESSMENT_REPORT', Path(__file__).parent.parent / "assessment_report.csv"))


def get_file_info(file_path: Path) -> Dict[str, Any]:
    """Get basic information about a file."""
    try:
        stat = file_path.stat()
        size_mb = stat.st_size / (1024 * 1024)

        # Determine file type
        mime_type, _ = mimetypes.guess_type(str(file_path))
        extension = file_path.suffix.lower()

        # Check if file is supported
        is_supported = extension in SUPPORTED_EXTENSIONS

        # Determine readability status
        if not is_supported:
            status = "Unsupported"
        elif size_mb > 100:  # Skip very large files
            status = "Too Large"
        elif not file_path.is_file():
            status = "Not a File"
        else:
            try:
                # Try to read first few bytes to check if file is accessible
                with open(file_path, 'rb') as f:
                    f.read(1024)
                status = "Readable"
            except (PermissionError, OSError):
                status = "Permission Denied"
            except Exception:
                status = "Corrupt"

        return {
            'file_path': str(file_path),
            'relative_path': str(file_path.relative_to(SOURCE_DIR)),
            'file_name': file_path.name,
            'extension': extension,
            'size_mb': round(size_mb, 3),
            'mime_type': mime_type or 'unknown',
            'status': status,
            'modified_date': datetime.fromtimestamp(stat.st_mtime).isoformat()
        }
    except Exception as e:
        return {
            'file_path': str(file_path),
            'relative_path': str(file_path.relative_to(SOURCE_DIR)) if SOURCE_DIR in file_path.parents else str(file_path),
            'file_name': file_path.name,
            'extension': file_path.suffix.lower() if hasattr(file_path, 'suffix') else '',
            'size_mb': 0,
            'mime_type': 'error',
            'status': f"Error: {str(e)}",
            'modified_date': ''
        }


def scan_directory(source_dir: Path) -> List[Dict[str, Any]]:
    """Recursively scan directory and collect file information."""
    files_info = []

    print(f"Scanning directory: {source_dir}")

    try:
        for root, dirs, files in os.walk(source_dir):
            root_path = Path(root)

            # Skip hidden directories and common non-content directories
            dirs[:] = [d for d in dirs if not d.startswith('.') and d not in ['__pycache__', 'node_modules', '.git']]

            for file in files:
                # Skip hidden files and common system files
                if file.startswith('.') or file in ['Thumbs.db', 'desktop.ini', '.DS_Store']:
                    continue

                file_path = root_path / file
                file_info = get_file_info(file_path)
                files_info.append(file_info)

                # Progress indicator
                if len(files_info) % 100 == 0:
                    print(f"Scanned {len(files_info)} files...")

    except Exception as e:
        print(f"Error scanning directory: {e}")

    return files_info


def generate_summary(files_info: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Generate summary statistics from the file information."""
    total_files = len(files_info)
    readable_files = sum(1 for f in files_info if f['status'] == 'Readable')
    total_size_mb = sum(f['size_mb'] for f in files_info)

    # Count by extension
    extensions = {}
    status_counts = {}

    for file_info in files_info:
        ext = file_info['extension']
        status = file_info['status']

        extensions[ext] = extensions.get(ext, 0) + 1
        status_counts[status] = status_counts.get(status, 0) + 1

    # Estimate processing time (rough estimate: 1 second per file + extra for images)
    image_extensions = {'.png', '.jpg', '.jpeg', '.tiff', '.tif', '.bmp', '.gif'}
    image_files = sum(1 for f in files_info if f['extension'] in image_extensions and f['status'] == 'Readable')

    estimated_time_minutes = (readable_files + image_files * 2) / 60  # Images take longer due to OCR

    return {
        'total_files': total_files,
        'readable_files': readable_files,
        'total_size_mb': round(total_size_mb, 2),
        'file_types': dict(sorted(extensions.items())),
        'status_breakdown': status_counts,
        'estimated_processing_time_minutes': round(estimated_time_minutes, 1)
    }


def save_assessment_report(files_info: List[Dict[str, Any]], summary: Dict[str, Any]):
    """Save the assessment report to CSV file."""

    # Save detailed file report
    with open(ASSESSMENT_REPORT, 'w', newline='', encoding='utf-8') as csvfile:
        if files_info:
            fieldnames = files_info[0].keys()
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(files_info)

    # Print summary to console
    print(f"\n{'='*60}")
    print("ASSESSMENT SUMMARY")
    print(f"{'='*60}")
    print(f"Total files found: {summary['total_files']}")
    print(f"Readable files: {summary['readable_files']}")
    print(f"Total size: {summary['total_size_mb']} MB")
    print(f"Estimated processing time: {summary['estimated_processing_time_minutes']} minutes")

    print(f"\nFile types breakdown:")
    for ext, count in list(summary['file_types'].items())[:10]:  # Show top 10
        print(f"  {ext or '(no extension)'}: {count} files")

    print(f"\nStatus breakdown:")
    for status, count in summary['status_breakdown'].items():
        print(f"  {status}: {count} files")

    print(f"\nDetailed report saved to: {ASSESSMENT_REPORT}")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(description="Assess files in the source directory")
    parser.add_argument("--source", type=str, help=f"Source directory path (default: {SOURCE_DIR})")
    args = parser.parse_args()

    source_dir = Path(args.source) if args.source else SOURCE_DIR

    if not source_dir.exists():
        print(f"Error: Source directory does not exist: {source_dir}")
        print(f"Please create the directory and add your files, or specify a different path with --source")
        return

    if not any(source_dir.iterdir()):
        print(f"Warning: Source directory is empty: {source_dir}")
        print("Add some files to the directory before running assessment.")
        return

    print("Starting file assessment...")
    files_info = scan_directory(source_dir)

    if not files_info:
        print("No files found to assess.")
        return

    summary = generate_summary(files_info)
    save_assessment_report(files_info, summary)


if __name__ == "__main__":
    main()
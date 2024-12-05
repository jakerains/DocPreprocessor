# Changelog

## [Unreleased]

### Added
- Enhanced CSS styling for better UI presentation
- Improved document output formatting
- Better error handling and user feedback
- Progress tracking with step-by-step updates
- Helpful information and usage guides
- File size display with human-readable format
- Multi-format download options for processed documents
- Improved error handling for JSON processing
- Additional format conversion options
- Better type checking and validation
- Debug logging for document processing
- Fallback methods for content extraction
- Character encoding detection
- Human-readable file size formatting
- OCR support for images in documents
- Image content extraction and processing
- Confidence scores for OCR results
- Structured JSON output for images and text
- Markdown formatting for OCR results
- Custom PageItem serialization
- Page-specific information in output formats
- Document structure preservation in JSON
- Image classification and description support
- Chart and diagram analysis
- Image caption extraction
- Confidence scores for image classifications
- Structured image metadata
- Enhanced image content organization

### Changed
- Updated warning suppressions for torch and easyocr
- Improved JSON and Markdown output formatting
- Enhanced preview functionality with better rendering
- Pinned PyTorch to version 2.0.1 for better compatibility
- Pinned torchvision to version 0.15.2
- Enhanced document output formatting
- Improved JSON structure and validation
- Added fallback content handling
- Better error messages for failed processing
- Enhanced CSS styling for a modern UI
- Updated tab styling with better visual feedback
- Improved document information display
- Enhanced JSON serialization for complex objects
- Better metadata handling in JSON output
- Improved image content organization in output formats
- Restructured document page handling
- Enhanced PageItem serialization logic
- Improved document structure representation
- Better image analysis integration
- Enhanced PictureItem handling
- Improved image annotation processing
- Better caption resolution
- Enhanced image classification display

### Fixed
- Resolved torch.classes warning with specific warning suppression
- Fixed messy UI presentation issues
- Improved text formatting and display
- Enhanced error handling with user-friendly messages
- Resolved NoneType JSON processing error
- Improved error handling for document conversion
- Added proper MIME type handling for downloads
- Fixed document attribute access issues
- Fixed tab styling inconsistencies
- Resolved content extraction issues with multiple fallback methods
- Fixed temporary file cleanup issues
- Fixed JSON serialization for PageItem objects
- Improved handling of non-serializable metadata
- Enhanced error handling for image processing
- Fixed PageItem serialization issues
- Resolved nested object serialization problems
- Improved handling of complex document structures
- Fixed PictureItem serialization
- Resolved image annotation extraction issues
- Fixed caption resolution problems
- Improved image classification handling
- Enhanced image content extraction
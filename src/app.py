from docling.document_converter import DocumentConverter
import streamlit as st
from pathlib import Path
import tempfile
import traceback
import humanize
import time
import json
import warnings
import torch
import easyocr
from docling.pipeline.standard_pdf_pipeline import StandardPdfPipeline
from docling.pipeline.simple_pipeline import SimplePipeline
from docling.backend.docling_parse_backend import DoclingParseDocumentBackend
from docling.backend.mspowerpoint_backend import MsPowerpointDocumentBackend

# More specific warning suppressions
warnings.filterwarnings('ignore', category=UserWarning, module='torch')
warnings.filterwarnings('ignore', category=UserWarning, module='easyocr')
warnings.filterwarnings('ignore', message='.*torch.classes.*')

# Configure the page with a cleaner theme
st.set_page_config(
    page_title="Sterling Document PreProcessor",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Modern UI styling
st.markdown("""
    <style>
    /* Modern color scheme */
    :root {
        --primary-color: #7C3AED;
        --secondary-color: #4F46E5;
        --background-color: #F9FAFB;
        --surface-color: #FFFFFF;
        --text-color: #1F2937;
        --success-color: #059669;
        --error-color: #DC2626;
    }
    
    /* Global styles */
    .stApp {
        background-color: var(--background-color);
    }
    
    .main {
        max-width: 1400px;
        padding: 2rem;
        margin: 0 auto;
    }
    
    /* Header styling */
    h1, h2, h3 {
        color: var(--text-color);
        font-weight: 600;
        letter-spacing: -0.025em;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(45deg, var(--primary-color), var(--secondary-color)) !important;
        color: white !important;
        border: none !important;
        padding: 0.75rem 1.5rem !important;
        border-radius: 8px !important;
        font-weight: 500 !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1) !important;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 8px -1px rgba(0, 0, 0, 0.15) !important;
    }
    
    /* Upload area styling */
    [data-testid="stFileUploader"] {
        background-color: var(--surface-color);
        border: 2px dashed var(--primary-color);
        border-radius: 12px;
        padding: 2rem;
        margin: 1rem 0;
        transition: all 0.3s ease;
    }
    
    [data-testid="stFileUploader"]:hover {
        border-color: var(--secondary-color);
        background-color: rgba(124, 58, 237, 0.05);
    }
    
    /* Progress bar styling */
    .stProgress > div > div > div {
        background: linear-gradient(45deg, var(--primary-color), var(--secondary-color));
        border-radius: 999px;
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        background-color: var(--surface-color);
        border-radius: 8px;
        border: 1px solid #E5E7EB;
        padding: 1rem;
        font-weight: 500;
    }
    
    .streamlit-expanderContent {
        background-color: var(--surface-color);
        border: 1px solid #E5E7EB;
        border-top: none;
        border-radius: 0 0 8px 8px;
        padding: 1.5rem;
    }
    
    /* Text area styling */
    .stTextArea textarea {
        font-family: 'SF Mono', 'Fira Code', 'Consolas', monospace;
        font-size: 14px;
        line-height: 1.6;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #E5E7EB;
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: transparent;
    }
    
    .stTabs [data-baseweb="tab"] {
        padding: 8px 16px;
        background-color: var(--surface-color);
        border-radius: 8px 8px 0 0;
        border: 1px solid #E5E7EB;
        border-bottom: none;
        position: relative;
    }
    
    .stTabs [data-baseweb="tab"]::after {
        content: "";
        position: absolute;
        bottom: 0;
        left: 0;
        right: 0;
        height: 2px;
        background-color: var(--primary-color);
        transform: scaleX(0);
        transform-origin: left;
        transition: transform 0.3s ease;
    }
    
    .stTabs [data-baseweb="tab"]:hover::after {
        transform: scaleX(1);
    }
    
    .stTabs [data-baseweb="tab"]:focus::after,
    .stTabs [data-baseweb="tab"].is-selected::after {
        transform: scaleX(1);
    }
    
    /* Success message styling */
    .success-message {
        color: var(--success-color);
        font-weight: bold;
        padding: 1rem;
        border-radius: 4px;
        background-color: #f1f8e9;
    }
    
    /* Error message styling */
    .error-message {
        color: var(--error-color);
        font-weight: bold;
        padding: 1rem;
        border-radius: 4px;
        background-color: #fef2f2;
    }
    
    /* Markdown styling */
    .stMarkdown {
        line-height: 1.6;
    }
    </style>
""", unsafe_allow_html=True)

# App header
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.title("📄 Sterling Document PreProcessor")
    st.markdown("""
        Transform your documents into LLM-ready data with ease. 
        Upload any supported document and get it processed in your preferred format.
    """)

# Initialize document converter
@st.cache_resource
def get_converter():
    from docling.datamodel.pipeline_options import PdfPipelineOptions, EasyOcrOptions
    from docling.document_converter import FormatOption, InputFormat
    
    # Configure OCR options
    ocr_options = EasyOcrOptions(
        lang=["en"],  # English OCR
        use_gpu=True  # Use GPU if available
    )
    
    # Configure pipeline options for PDF
    pdf_pipeline_options = PdfPipelineOptions(
        do_ocr=True,  # Enable OCR
        ocr_options=ocr_options,
        do_table_structure=True,  # Enable table structure analysis
        generate_picture_images=True,  # Enable image extraction
        generate_page_images=True,     # Enable page image extraction
        images_scale=1.5,              # Higher quality images
    )
    
    # Create format options dictionary
    format_options = {
        InputFormat.PDF: FormatOption(
            pipeline_cls=StandardPdfPipeline,
            pipeline_options=pdf_pipeline_options,
            backend=DoclingParseDocumentBackend
        ),
        InputFormat.PPTX: FormatOption(
            pipeline_cls=SimplePipeline,  # PowerPoint uses SimplePipeline
            backend=MsPowerpointDocumentBackend  # PowerPoint-specific backend
        )
    }
    
    return DocumentConverter(format_options=format_options)

doc_converter = get_converter()

# Sidebar with options
with st.sidebar:
    st.header("Processing Options")
    
    # Output format selection
    output_format = st.selectbox(
        "Output Format",
        ["Text", "JSON", "Markdown"],
        index=0,
        help="Choose how you want your processed document to be formatted"
    )
    
    # Advanced options
    with st.expander("Advanced Options"):
        chunk_size = st.number_input(
            "Chunk Size (characters)",
            min_value=100,
            max_value=10000,
            value=1000,
            step=100,
            help="Set the size of text chunks for processing"
        )
        
        preserve_formatting = st.checkbox(
            "Preserve Formatting",
            value=True,
            help="Keep original document formatting in the output"
        )
        
        extract_metadata = st.checkbox(
            "Extract Metadata",
            value=True,
            help="Include document metadata in the output"
        )

# Main content area
main_col1, main_col2 = st.columns([2, 3])

with main_col1:
    st.subheader("Upload Document")
    
    # File uploader with supported formats
    supported_formats = ["txt", "pdf", "docx", "pptx", "xlsx", "html", "md"]
    st.markdown(f"**Supported formats:** {', '.join(f'*.{fmt}' for fmt in supported_formats)}")
    
    uploaded_file = st.file_uploader(
        "Drop your file here or click to browse",
        type=supported_formats,
        help="Select a document to process"
    )

def serialize_picture_item(picture_item, doc):
    """Helper function to serialize PictureItem objects"""
    if not picture_item:
        return None
    
    result = {
        'type': 'picture',
        'annotations': [],
        'captions': [],
        'location': None,
        'image_info': None
    }
    
    # Get image data
    try:
        # Try to get the image directly from the PictureItem
        if hasattr(picture_item, 'image') and picture_item.image:
            if hasattr(picture_item.image, '_pil'):
                img = picture_item.image._pil
                result['image_info'] = {
                    'size': {'width': img.width, 'height': img.height},
                    'mode': img.mode,
                    'format': img.format
                }
        # If that fails, try getting it through get_image
        elif hasattr(picture_item, 'get_image'):
            img = picture_item.get_image(doc)
            if img:
                result['image_info'] = {
                    'size': {'width': img.width, 'height': img.height},
                    'mode': img.mode,
                    'format': img.format
                }
    except Exception as e:
        st.warning(f"Could not get image data: {str(e)}")
    
    # Handle location
    if hasattr(picture_item, 'prov') and picture_item.prov:
        prov = picture_item.prov[0]  # Get first provenance
        result['location'] = {
            'page_no': prov.page_no,
            'bbox': prov.bbox.as_tuple() if hasattr(prov.bbox, 'as_tuple') else prov.bbox
        }
    
    # Handle annotations
    if hasattr(picture_item, 'annotations'):
        for annotation in picture_item.annotations:
            if hasattr(annotation, 'kind'):
                if annotation.kind == 'classification':
                    # Handle classification data
                    classes = []
                    for pred_class in annotation.predicted_classes:
                        classes.append({
                            'class_name': pred_class.class_name,
                            'confidence': pred_class.confidence
                        })
                    result['annotations'].append({
                        'kind': 'classification',
                        'provenance': annotation.provenance,
                        'predicted_classes': classes
                    })
                elif annotation.kind == 'description':
                    # Handle description data
                    result['annotations'].append({
                        'kind': 'description',
                        'text': annotation.text,
                        'provenance': annotation.provenance
                    })
    
    # Handle captions
    if hasattr(picture_item, 'captions'):
        for caption in picture_item.captions:
            if isinstance(caption, str):
                result['captions'].append(caption)
            elif hasattr(caption, 'text'):
                result['captions'].append(caption.text)
    
    return result

def format_output(doc, format_type):
    """Format document output based on selected format type"""
    try:
        # Debugging: List all attributes of the doc
        st.write("Debug - Document attributes:", dir(doc))
        
        if not doc:
            st.error("Document object is None")
            return None

        def serialize_page_item(page_item):
            """Helper function to serialize PageItem objects"""
            if not page_item:
                return None
            
            # Convert Size object to dict
            size_dict = None
            if hasattr(page_item, 'size'):
                size = page_item.size
                size_dict = {
                    'width': getattr(size, 'width', 0),
                    'height': getattr(size, 'height', 0)
                }
            
            # Convert ImageRef to dict
            image_dict = None
            if hasattr(page_item, 'image') and page_item.image:
                image = page_item.image
                image_dict = {
                    'mode': getattr(image, 'mode', None),
                    'format': getattr(image, 'format', None)
                }
            
            return {
                'page_no': getattr(page_item, 'page_no', None),
                'size': size_dict,
                'image': image_dict
            }

        def serialize_item(item):
            """Helper function to serialize document items"""
            if not item:
                return None
            
            # Handle PageItem specifically
            if hasattr(item, '__class__'):
                if item.__class__.__name__ == 'PageItem':
                    return serialize_page_item(item)
                elif item.__class__.__name__ == 'PictureItem':
                    return serialize_picture_item(item, doc)
            
            # Handle other items
            if hasattr(item, '__dict__'):
                result = {}
                for key, value in item.__dict__.items():
                    # Skip private attributes
                    if key.startswith('_'):
                        continue
                    
                    # Handle nested objects
                    if hasattr(value, '__dict__'):
                        result[key] = serialize_item(value)
                    # Handle lists/iterables
                    elif isinstance(value, (list, tuple)):
                        result[key] = [serialize_item(v) if hasattr(v, '__dict__') else str(v) for v in value]
                    # Handle basic types
                    else:
                        try:
                            json.dumps({key: value})
                            result[key] = value
                        except (TypeError, ValueError):
                            result[key] = str(value)
                return result
            return str(item)

        # Process document content
        text_content = []
        pages_info = []
        pictures_info = []
        
        # Extract pages information
        if hasattr(doc, 'pages'):
            for page_no, page in doc.pages.items():
                page_info = serialize_page_item(page)
                if page_info:
                    pages_info.append(page_info)
        
        # Extract text and picture content
        if hasattr(doc, 'items') and doc.items:
            for item in doc.items:
                if hasattr(item, '__class__'):
                    if item.__class__.__name__ == 'PictureItem':
                        picture_info = serialize_picture_item(item, doc)
                        if picture_info:
                            pictures_info.append(picture_info)
                    elif hasattr(item, 'text') and item.text:
                        text_content.append(str(item.text).strip())
        elif hasattr(doc, 'text') and doc.text:
            text_content.append(str(doc.text).strip())
        elif hasattr(doc, 'export_to_text'):
            text_content.append(str(doc.export_to_text()).strip())
        elif hasattr(doc, 'content') and doc.content:
            text_content.append(str(doc.content).strip())
        
        # Combine all text content
        text = '\n'.join(text_content)
        
        if not text and not pages_info and not pictures_info:
            st.error("No content could be extracted from the document")
            return None
        
        # Format based on type
        if format_type == "Text":
            # Combine text, page information, and picture information
            output_parts = []
            if text:
                output_parts.append(text)
            
            if pages_info:
                output_parts.append("\n=== Document Pages ===")
                for page in pages_info:
                    output_parts.append(f"\nPage {page['page_no']}:")
                    if page['size']:
                        output_parts.append(f"Size: {page['size']['width']}x{page['size']['height']}")
            
            if pictures_info:
                output_parts.append("\n=== Images and Charts ===")
                for idx, pic in enumerate(pictures_info, 1):
                    output_parts.append(f"\nImage {idx}:")
                    
                    # Add location information
                    if pic['location']:
                        output_parts.append(f"Location: Page {pic['location']['page_no']}")
                    
                    # Add image information
                    if pic['image_info']:
                        output_parts.append(f"Size: {pic['image_info']['size']['width']}x{pic['image_info']['size']['height']}")
                        output_parts.append(f"Format: {pic['image_info']['format']}")
                    
                    # Add captions
                    if pic['captions']:
                        output_parts.append("Caption: " + " ".join(pic['captions']))
                    
                    # Add annotations
                    if pic['annotations']:
                        for annotation in pic['annotations']:
                            if annotation['kind'] == 'description':
                                output_parts.append(f"Description: {annotation['text']}")
                            elif annotation['kind'] == 'classification':
                                output_parts.append("Classifications:")
                                for pred_class in annotation['predicted_classes']:
                                    output_parts.append(f"- {pred_class['class_name']} (confidence: {pred_class['confidence']:.2f})")
            
            return '\n'.join(output_parts)
        
        elif format_type == "JSON":
            # Create a JSON-serializable dictionary
            doc_dict = {
                'text_content': text if text else '',
                'pages': pages_info,
                'images': pictures_info,
                'metadata': {
                    'filename': str(getattr(doc, 'name', '')),
                    'type': str(getattr(doc.origin, 'mimetype', '')) if hasattr(doc, 'origin') else '',
                    'total_pages': len(pages_info),
                    'total_images': len(pictures_info),
                    'processed_at': time.strftime('%Y-%m-%d %H:%M:%S')
                }
            }
            
            # Add any additional metadata
            if hasattr(doc, 'metadata'):
                try:
                    meta_dict = serialize_item(doc.metadata)
                    doc_dict['metadata'].update(meta_dict)
                except Exception as meta_error:
                    st.warning(f"Could not process some metadata: {str(meta_error)}")
            
            try:
                return json.dumps(doc_dict, indent=2, ensure_ascii=False)
            except TypeError as json_error:
                st.error(f"JSON serialization error: {str(json_error)}")
                # Attempt to serialize with str conversion
                doc_dict_safe = serialize_item(doc_dict)
                return json.dumps(doc_dict_safe, indent=2, ensure_ascii=False)
        
        elif format_type == "Markdown":
            # Create markdown with text, page information, and picture information
            md_parts = []
            if text:
                md_parts.append(f"# {getattr(doc, 'name', 'Document')}\n")
                md_parts.append(text)
            
            if pages_info:
                md_parts.append("\n## Document Pages\n")
                for page in pages_info:
                    md_parts.append(f"\n### Page {page['page_no']}\n")
                    if page['size']:
                        md_parts.append(f"**Size:** {page['size']['width']}x{page['size']['height']}\n")
            
            if pictures_info:
                md_parts.append("\n## Images and Charts\n")
                for idx, pic in enumerate(pictures_info, 1):
                    md_parts.append(f"\n### Image {idx}\n")
                    
                    # Add location information
                    if pic['location']:
                        md_parts.append(f"**Location:** Page {pic['location']['page_no']}\n")
                    
                    # Add image information
                    if pic['image_info']:
                        md_parts.append(f"**Size:** {pic['image_info']['size']['width']}x{pic['image_info']['size']['height']}\n")
                        md_parts.append(f"**Format:** {pic['image_info']['format']}\n")
                    
                    # Add captions
                    if pic['captions']:
                        md_parts.append(f"**Caption:** {' '.join(pic['captions'])}\n")
                    
                    # Add annotations
                    if pic['annotations']:
                        for annotation in pic['annotations']:
                            if annotation['kind'] == 'description':
                                md_parts.append(f"**Description:** {annotation['text']}\n")
                            elif annotation['kind'] == 'classification':
                                md_parts.append("**Classifications:**\n")
                                for pred_class in annotation['predicted_classes']:
                                    md_parts.append(f"- {pred_class['class_name']} (confidence: {pred_class['confidence']:.2f})\n")
            
            return '\n'.join(md_parts)
        
        else:
            raise ValueError(f"Unsupported format: {format_type}")
    
    except Exception as e:
        st.error(f"Error formatting output: {str(e)}")
        st.error(f"Traceback: {traceback.format_exc()}")
        return None

# Process the document
if uploaded_file is not None:
    try:
        with main_col2:
            st.subheader("Processing Results")
            
            # Show file info
            file_size = len(uploaded_file.getvalue())
            st.info(f"📁 File: {uploaded_file.name} ({humanize.naturalsize(file_size)})")
            
            # Process with progress bar
            with st.spinner("Processing document..."):
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Create temporary file
                with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as temp_file:
                    temp_file.write(uploaded_file.getvalue())
                    temp_file_path = Path(temp_file.name)
                
                # Processing steps
                steps = ["Analyzing document", "Extracting content", "Formatting output"]
                for i, step in enumerate(steps):
                    status_text.text(f"Step {i+1}/{len(steps)}: {step}")
                    progress = (i * 33) + 1
                    progress_bar.progress(progress)
                    time.sleep(0.5)  # Simulate processing time
                
                # Convert document with debug info
                st.write("Debug - Starting document conversion...")
                conv_result = doc_converter.convert(temp_file_path)
                
                if not conv_result:
                    st.error("Document conversion failed - no result returned")
                    st.stop()  # Halt execution
                
                doc = conv_result.document
                if not doc:
                    st.error("Document conversion failed - no document in result")
                    st.stop()  # Halt execution
                
                st.write(f"Debug - Document type: {type(doc)}")
                st.write(f"Debug - Conversion result type: {type(conv_result)}")
                
                processed_content = format_output(doc, output_format)
                if not processed_content:
                    st.error("Failed to process document content")
                    st.stop()  # Halt execution
                
                progress_bar.progress(100)
                status_text.text("Processing complete!")
                
                # Success message
                st.success("✅ Document processed successfully!")
                
                # Document information
                with st.expander("Document Information", expanded=True):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown("**Basic Information**")
                        st.json({
                            "File Type": str(temp_file_path.suffix),
                            "Size": humanize.naturalsize(file_size),
                            "Pages": getattr(doc, 'pages', 'N/A'),
                            "Characters": len(doc.text) if hasattr(doc, 'text') else 0,
                        })
                    
                    with col2:
                        st.markdown("**Processing Details**")
                        st.json({
                            "Output Format": output_format,
                            "Chunk Size": chunk_size,
                            "Formatting Preserved": preserve_formatting,
                            "Metadata Extracted": extract_metadata
                        })
                
                # Display processed content
                st.subheader("Processed Content")
                
                # Add content preview options
                preview_tab, raw_tab = st.tabs(["Preview", "Raw Output"])
                
                with preview_tab:
                    if output_format == "Text":
                        st.text_area(
                            "Content Preview",
                            processed_content,
                            height=400,
                            help="Processed text content with preserved formatting"
                        )
                    elif output_format == "JSON":
                        try:
                            # Ensure we have valid JSON before trying to display it
                            json_content = json.loads(processed_content) if isinstance(processed_content, str) else None
                            if json_content:
                                st.json(json_content, expanded=False)
                            else:
                                st.error("Invalid JSON content")
                        except json.JSONDecodeError:
                            st.error("Invalid JSON format")
                    else:  # Markdown
                        st.markdown(
                            processed_content,
                            help="Rendered markdown content with formatting"
                        )
                
                with raw_tab:
                    st.code(processed_content, language="python" if output_format == "JSON" else "markdown")
                
                # Add download functionality with proper error handling
                if processed_content:
                    try:
                        # Determine file extension and mime type
                        file_extensions = {
                            "Text": ".txt",
                            "JSON": ".json",
                            "Markdown": ".md"
                        }
                        mime_types = {
                            "Text": "text/plain",
                            "JSON": "application/json",
                            "Markdown": "text/markdown"
                        }
                        
                        output_extension = file_extensions.get(output_format, ".txt")
                        mime_type = mime_types.get(output_format, "text/plain")
                        
                        # Create download button
                        st.download_button(
                            label=f"📥 Download as {output_format}",
                            data=processed_content,
                            file_name=f"processed_{Path(uploaded_file.name).stem}{output_extension}",
                            mime=mime_type,
                            key=f"download_{output_format.lower()}"
                        )
                        
                        # Add additional format options
                        with st.expander("Download in Other Formats"):
                            for fmt in ["Text", "JSON", "Markdown"]:
                                if fmt != output_format:
                                    alt_content = format_output(doc, fmt)
                                    if alt_content:
                                        alt_extension = file_extensions.get(fmt, ".txt")
                                        alt_mime = mime_types.get(fmt, "text/plain")
                                        st.download_button(
                                            label=f"📥 Download as {fmt}",
                                            data=alt_content,
                                            file_name=f"processed_{Path(uploaded_file.name).stem}{alt_extension}",
                                            mime=alt_mime,
                                            key=f"download_{fmt.lower()}"
                                        )
                                    
                    except Exception as e:
                        st.error(f"Error creating download button: {str(e)}")
    
    except Exception as e:
        st.error(f"Error during document conversion: {str(e)}")
        st.error(f"Traceback: {traceback.format_exc()}")
    
    finally:
        # Clean up temporary file
        if 'temp_file_path' in locals():
            try:
                temp_file_path.unlink()
            except Exception as cleanup_error:
                st.error(f"Error deleting temporary file: {cleanup_error}")

else:
    with main_col2:
        # Show helpful information when no file is uploaded
        st.info("👈 Upload a document to get started!")
        
        with st.expander("How to use"):
            st.markdown("""
                1. **Select a document** using the file uploader on the left.
                2. **Choose your output format** from the sidebar:
                   - **Text:** Plain text output.
                   - **JSON:** Structured data format.
                   - **Markdown:** Formatted text with preserved structure.
                3. **Adjust advanced options** if needed:
                   - **Chunk Size:** Control text segmentation.
                   - **Preserve Formatting:** Keep original styling.
                   - **Extract Metadata:** Include document properties.
                4. **Wait for processing** to complete.
                5. **Preview and download** your processed document.
            """)
            
        with st.expander("Features"):
            st.markdown("""
                - **Multiple Format Support:** Process various document types.
                - **Smart Processing:** Intelligent text extraction and formatting.
                - **Preview Options:** View content before downloading.
                - **Advanced Settings:** Fine-tune the processing.
                - **Error Handling:** Clear feedback and troubleshooting tips.
            """)
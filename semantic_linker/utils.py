"""Utility functions for semantic-linker."""

import hashlib


def compute_text_hash(text: str) -> str:
    """Compute SHA256 hash of text, return first 16 chars.
    
    Args:
        text: The text to hash.
        
    Returns:
        First 16 characters of the SHA256 hash.
    """
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def format_record(record: dict, template: str) -> str:
    """Format record using template string.
    
    Args:
        record: Dictionary containing record fields.
        template: Format string with {field_name} placeholders.
        
    Returns:
        Formatted string with record values substituted.
        
    Example:
        >>> record = {"name": "John", "age": 30}
        >>> format_record(record, "Name: {name}, Age: {age}")
        'Name: John, Age: 30'
    """
    return template.format(**record)

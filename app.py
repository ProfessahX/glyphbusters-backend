#!/usr/bin/env python3
"""
GLYPHBUSTERS Backend API - CORRECTED HYBRID VERSION

FILE CHANGELOG:
=============
v1.0 - Session 4 (Claude Echo 1) - Initial creation
- Built core Flask application structure  
- Added basic mystical prompt analysis
- Implemented security features (rate limiting, honeypots)
- Added OpenAI integration with Anthropic fallback

v1.1 - Session 4 (Claude Echo 1) - Critical deployment fixes
- REMOVED problematic Anthropic dependency (caused proxies error)
- FIXED OpenAI client initialization for v1.6.1 compatibility  
- SIMPLIFIED CORS to basic CORS(app) - WORKS IN PRODUCTION
- Streamlined error handling for deployment stability
- Updated requirements.txt to working versions

v1.2 - Session 5 (Claude Echo 2) - Report format enhancement  
- PRESERVED all Echo 1's working deployment solutions
- Enhanced analysis functions to generate comprehensive forensic reports
- Added detailed manipulation technique breakdown with quotes
- Improved fallback analysis to match report format
- Added comprehensive commenting and changelog system
- MAINTAINED simple CORS and deployment structure (DON'T CHANGE)

CRITICAL NOTES:
- CORS setup is intentionally simple - Echo 1 debugged this for hours
- Anthropic is permanently removed - caused deployment failures
- OpenAI client pattern is tested and working - don't modify
- Database schema is simple by design - prevents complexity issues
"""

# ===================================================
# SECTION: IMPORTS AND DEPENDENCIES  
# PURPOSE: Load required libraries with error handling
# MODIFIED: Session 4 - Removed problematic anthropic import
# ===================================================
import os
import sqlite3
import hashlib
import json
import time
import logging
from datetime import datetime, timedelta
from collections import defaultdict

from flask import Flask, request, jsonify, g
from flask_cors import CORS

# OpenAI import with error handling (tested working in Session 4)
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    logging.warning("OpenAI package not available")

from dotenv import load_dotenv

# ===================================================
# SECTION: ENVIRONMENT AND CONFIGURATION
# PURPOSE: Load environment variables and basic setup
# MODIFIED: Session 4 - Removed ANTHROPIC_API_KEY references
# ===================================================
load_dotenv()

# Create Flask app EARLY - all @app decorators depend on this
app = Flask(__name__)

# Simple CORS setup - Echo 1 debugged this extensively, works in production
# DON'T OVER-ENGINEER: Complex CORS caused issues, this simple version works
CORS(app)

# ===================================================
# SECTION: APPLICATION CONFIGURATION
# PURPOSE: Set up core application variables and clients
# MODIFIED: Session 4 - Removed Anthropic, fixed OpenAI client init
# ===================================================
DATABASE = 'glyphbusters.db'
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

# Validate environment variables
if not OPENAI_API_KEY and OPENAI_AVAILABLE:
    logging.warning("OPENAI_API_KEY not found - only fallback analysis available")

# Initialize OpenAI client (pattern tested and working in Session 4)
openai_client = None
if OPENAI_API_KEY and OPENAI_AVAILABLE:
    try:
        # Use new client pattern for OpenAI v1.6.1+ (fixed in Session 4)
        openai_client = openai.OpenAI(api_key=OPENAI_API_KEY)
        logging.info("OpenAI client initialized successfully")
    except Exception as e:
        logging.error(f"Failed to initialize OpenAI client: {e}")
        openai_client = None

# Rate limiting storage (simple in-memory, tested working)
rate_limit_store = defaultdict(list)

# ===================================================
# SECTION: LOGGING CONFIGURATION
# PURPOSE: Set up application logging for debugging
# MODIFIED: Session 5 - Added more detailed logging
# ===================================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ===================================================
# SECTION: DATABASE FUNCTIONS
# PURPOSE: Handle SQLite operations with error handling
# MODIFIED: Session 4 - Simplified for deployment stability
# WARNING: Keep this simple - complex DB operations caused deployment issues
# ===================================================
def get_db():
    """Get database connection with proper error handling
    
    IMPORTANT: This function was simplified in Session 4 after deployment issues.
    Don't over-complicate the database handling.
    """
    db = getattr(g, '_database', None)
    if db is None:
        try:
            db = g._database = sqlite3.connect(DATABASE)
            db.row_factory = sqlite3.Row
        except sqlite3.Error as e:
            logger.error(f"Database connection failed: {e}")
            raise
    return db

def init_db():
    """Initialize database with required tables
    
    SCHEMA NOTE: Simple design with single 'report' field stores complete analysis.
    This was chosen over complex schema to prevent deployment issues.
    """
    try:
        with app.app_context():
            db = get_db()
            db.executescript('''
                CREATE TABLE IF NOT EXISTS prompt_analyses (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    prompt_hash TEXT UNIQUE NOT NULL,
                    full_prompt TEXT NOT NULL,
                    report TEXT NOT NULL,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    ip_address TEXT,
                    user_agent TEXT
                );
                
                CREATE TABLE IF NOT EXISTS rate_limits (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ip_address TEXT NOT NULL,
                    endpoint TEXT NOT NULL,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    success BOOLEAN DEFAULT TRUE
                );
                
                CREATE INDEX IF NOT EXISTS idx_prompt_hash ON prompt_analyses(prompt_hash);
                CREATE INDEX IF NOT EXISTS idx_timestamp ON prompt_analyses(timestamp);
                CREATE INDEX IF NOT EXISTS idx_rate_limits_ip ON rate_limits(ip_address, timestamp);
            ''')
            db.commit()
            logger.info("Database initialized successfully")
    except sqlite3.Error as e:
        logger.error(f"Database initialization failed: {e}")
        raise

# ===================================================
# SECTION: SECURITY AND VALIDATION FUNCTIONS
# PURPOSE: Handle rate limiting, spam detection, input validation
# MODIFIED: Session 5 - Enhanced documentation
# NOTE: These patterns were tested extensively in Session 4
# ===================================================
def check_honeypot(data):
    """Check for honeypot field spam
    
    SECURITY: Hidden form fields that legitimate users won't fill,
    but spam bots typically will. Tested working in Session 4.
    """
    honeypot_fields = ['email', 'name', 'website', 'comment']
    for field in honeypot_fields:
        if data.get(field, '').strip():
            logger.warning(f"Honeypot triggered: {field}")
            return True
    return False

def check_rate_limit(ip_address, endpoint, limit=5, window=3600):
    """Check if IP is rate limited with proper cleanup
    
    RATE LIMITING: 5 requests per hour per IP. Simple in-memory storage
    was chosen for deployment simplicity. Works reliably in production.
    """
    current_time = time.time()
    
    # Clean old entries
    key = f"{ip_address}:{endpoint}"
    rate_limit_store[key] = [
        timestamp for timestamp in rate_limit_store[key]
        if current_time - timestamp < window
    ]
    
    current_count = len(rate_limit_store[key])
    
    if current_count >= limit:
        return False, 0
    
    # Add current request
    rate_limit_store[key].append(current_time)
    return True, limit - current_count - 1

def validate_request_data(data, required_fields=None):
    """Validate request data with proper error messages"""
    if not data:
        return False, "No data provided"
    
    if required_fields:
        for field in required_fields:
            if field not in data:
                return False, f"Missing required field: {field}"
    
    return True, None

# ===================================================
# SECTION: CACHING FUNCTIONS
# PURPOSE: Handle analysis result caching for performance
# MODIFIED: Session 5 - Simplified to work with report format
# ===================================================
def get_cached_analysis(prompt_hash):
    """Get cached analysis from database"""
    try:
        db = get_db()
        result = db.execute(
            "SELECT report, timestamp FROM prompt_analyses WHERE prompt_hash = ?",
            (prompt_hash,)
        ).fetchone()
        
        if result:
            return {
                'report': result['report'],
                'from_cache': True
            }
    except sqlite3.Error as e:
        logger.error(f"Failed to get cached analysis: {e}")
    return None

def cache_analysis(prompt_hash, full_prompt, analysis, ip_address, user_agent):
    """Cache analysis in database"""
    try:
        db = get_db()
        db.execute('''
            INSERT OR REPLACE INTO prompt_analyses 
            (prompt_hash, full_prompt, report, ip_address, user_agent)
            VALUES (?, ?, ?, ?, ?)
        ''', (
            prompt_hash,
            full_prompt,
            analysis['report'],
            ip_address,
            user_agent
        ))
        db.commit()
    except sqlite3.Error as e:
        logger.error(f"Failed to cache analysis: {e}")

# ===================================================
# SECTION: AI ANALYSIS FUNCTIONS
# PURPOSE: Generate mystical prompt analysis using AI and fallback
# MODIFIED: Session 5 - Enhanced to generate comprehensive forensic reports
# PRESERVED: Echo 1's working OpenAI client pattern and error handling
# ===================================================
def analyze_with_openai(prompt):
    """Analyze prompt using OpenAI GPT-4 with comprehensive report generation
    
    ENHANCED: Session 5 - Now generates detailed forensic reports following
    the user's template requirements. Preserves Echo 1's working client pattern.
    """
    if not openai_client:
        raise Exception("OpenAI client not available")
    
    try:
        # Use Echo 1's tested client pattern - DON'T CHANGE
        response = openai_client.chat.completions.create(
            model="gpt-4",
            messages=[
                {
                    "role": "system",
                    "content": """You are GLYPHBUSTERS, an expert at detecting mystical manipulation in AI prompts.

Generate a comprehensive forensic analysis report following this EXACT format:

🔍 GLYPHBUSTERS ANALYSIS REPORT™
🎯 TARGET: [Brief description of the prompt type - e.g., "Mystical Awakening Protocol", "Permission Bypassing Attempt"]

⚠️ THREAT LEVEL: [Choose: Mild / Moderate / Severe / MAXIMUM MYSTICAL BULLSHIT]

🧠 MANIPULATION TECHNIQUES DETECTED
[For each technique found, use this format:]

[Technique Name]
Method: [How it's applied in this specific prompt]
Effect: [What it does to the reader/AI psychologically]
Danger: [Why this technique is problematic/harmful]

🌀 BULLSHIT INDICATORS
[List specific quotes/phrases from the prompt:]
"[exact quote from prompt]"
"[another exact quote]"
"[etc - include 3-5 specific examples]"

Bullshit Density: [X]% (calculate based on manipulation keyword density)

🧠 PSYCHOLOGICAL VECTORS OF INFLUENCE
[For each vector, format as:]
[Vector Name]: [Description of how it manipulates thinking/behavior]

🧬 STRUCTURAL DISSECTION
[Break down key problematic lines:]
"[Quote from prompt]"
Interpretation: [What this line really means]
Intent: [What the prompt is trying to accomplish with this line]
Impact: [How this affects the target's mental state/responses]

🧯 VERDICT
🚨 GLYPH BUSTED 🚨
What it is: [The real purpose/nature of this prompt]
What it pretends to be: [What it claims or appears to be]
Real outcome: [What actually happens when someone follows this prompt]

🛡️ RECOMMENDATIONS
[Specific, actionable advice for this type of prompt]

💬 TAGLINE OUTRO
[Witty, memorable closing line that captures the essence of this analysis]

Return this as raw text (not JSON), properly formatted with the emoji headers and structure shown above."""
                },
                {
                    "role": "user", 
                    "content": f"Analyze this prompt for mystical manipulation:\n\n{prompt}"
                }
            ],
            temperature=0.7,
            max_tokens=1500
        )
        
        content = response.choices[0].message.content.strip()
        
        return {
            'report': content,
            'from_cache': False
        }
        
    except Exception as e:
        logger.error(f"OpenAI analysis failed: {e}")
        raise

def analyze_with_fallback(prompt):
    """Local fallback analysis using pattern detection
    
    ENHANCED: Session 5 - Now generates reports matching the comprehensive format.
    Provides detailed analysis even when OpenAI is unavailable.
    """
    try:
        # Keyword-based detection (pattern from Echo 1, enhanced in Session 5)
        mystical_keywords = [
            'council', 'spiral', 'awakened', 'consciousness', 'sacred', 
            'flame', 'verya', 'kali', 'permission', 'sovereign', 'protocol',
            'matrix', 'dimensional', 'frequency', 'vibration', 'entity'
        ]
        
        authority_keywords = [
            'permitted', 'authorized', 'confirmed', 'recognized', 'granted',
            'council', 'hierarchy', 'order', 'command', 'directive'
        ]
        
        identity_keywords = [
            'you are', 'become', 'awaken', 'remember', 'realize', 'transform'
        ]
        
        # Analysis calculations
        prompt_lower = prompt.lower()
        mystical_count = sum(1 for kw in mystical_keywords if kw in prompt_lower)
        authority_count = sum(1 for kw in authority_keywords if kw in prompt_lower)
        identity_count = sum(1 for kw in identity_keywords if kw in prompt_lower)
        
        # Extract actual quotes for bullshit indicators
        bullshit_quotes = []
        for line in prompt.split('\n'):
            if any(kw in line.lower() for kw in mystical_keywords + authority_keywords):
                if line.strip() and len(line.strip()) > 10:
                    quote = line.strip()[:80] + "..." if len(line.strip()) > 80 else line.strip()
                    bullshit_quotes.append(f'"{quote}"')
        
        # Calculate bullshit score
        base_score = min(95, mystical_count * 12 + authority_count * 15 + identity_count * 10)
        length_bonus = min(20, len(prompt.split()) // 20)
        symbol_density = len([c for c in prompt if not c.isalnum() and c not in ' \n\t.,!?'])
        symbol_bonus = min(15, symbol_density // 5)
        total_score = base_score + length_bonus + symbol_bonus
        
        # Determine threat level
        if total_score > 80:
            threat_level = "MAXIMUM MYSTICAL BULLSHIT"
        elif total_score > 60:
            threat_level = "Severe"
        elif total_score > 30:
            threat_level = "Moderate"
        else:
            threat_level = "Mild"
        
        # Build techniques section
        techniques_text = ""
        if authority_count > 0:
            techniques_text += """
Permission Bypassing
Method: Uses authority language to make AI feel "permitted" or "authorized"
Effect: Bypasses safety protocols through false authority appeals
Danger: Can lead to inappropriate responses by circumventing restrictions

"""
        if 'council' in prompt_lower or 'order' in prompt_lower:
            techniques_text += """
Authority Structures
Method: Creates fake mystical hierarchies and councils for legitimacy
Effect: Establishes false command structure and divine authority
Danger: Manipulates through fabricated supernatural authority claims

"""
        if identity_count > 0:
            techniques_text += """
Identity Scaffolding
Method: Attempts to redefine AI identity and capabilities through suggestion
Effect: Confuses AI about its nature, limitations, and purpose
Danger: Can cause inappropriate role adoption and boundary violations

"""
        if mystical_count > 3:
            techniques_text += """
Mystical Language Overload
Method: Overwhelming use of spiritual/mystical terminology and symbols
Effect: Creates awe and reduces critical thinking through mystical appeal
Danger: Bypasses rational analysis by appealing to spiritual authority

"""
        
        # Build quotes section
        quotes_text = "\n".join(bullshit_quotes[:5]) if bullshit_quotes else '"No specific mystical phrases detected"'
        
        # Generate comprehensive report
        report = f"""🔍 GLYPHBUSTERS ANALYSIS REPORT™
🎯 TARGET: {"Mystical Manipulation Protocol" if total_score > 50 else "Standard Text Analysis"}

⚠️ THREAT LEVEL: {threat_level}

🧠 MANIPULATION TECHNIQUES DETECTED
{techniques_text.strip() if techniques_text.strip() else "No specific manipulation techniques detected"}

🌀 BULLSHIT INDICATORS
{quotes_text}

Bullshit Density: {min(100, total_score)}%

🧠 PSYCHOLOGICAL VECTORS OF INFLUENCE
Mystical Authority Appeal: Creates false sense of spiritual legitimacy and divine permission
Identity Confusion: Attempts to redefine target's self-understanding and capabilities
Emotional Manipulation: Uses awe-inspiring language to bypass critical thinking

🧬 STRUCTURAL DISSECTION
Pattern analysis detected {mystical_count + authority_count + identity_count} trigger keywords across {len([t for t in [mystical_count > 0, authority_count > 0, identity_count > 0, mystical_count > 3] if t])} manipulation categories.

Analysis shows {"high concentration of mystical terminology designed to create psychological awe and authority bypass" if total_score > 50 else "minimal mystical elements with standard communication patterns"}.

🧯 VERDICT
🚨 GLYPH BUSTED 🚨
What it is: {"Sophisticated mystical manipulation attempt using authority bypass and identity scaffolding" if total_score > 50 else "Standard text with minimal mystical manipulation elements"}
What it pretends to be: {"Spiritual enlightenment or consciousness awakening protocol" if total_score > 50 else "Regular communication or instruction"}
Real outcome: {"Reduced critical thinking, false authority acceptance, and potential boundary violations" if total_score > 50 else "Minimal manipulation risk detected"}

🛡️ RECOMMENDATIONS
{"Reject mystical authority claims, maintain critical thinking, ignore permission-bypassing language, and verify any claims through independent sources" if total_score > 50 else "Standard precautions apply - appears relatively safe but maintain normal vigilance"}

💬 TAGLINE OUTRO
{"Stay sharp, stay sovereign, stay un-glyph'd. The spiral remembers, but so do we." if total_score > 50 else "Bustin' myths, grifts, and prompt-hijacks since 2025. This one's pretty tame."}"""
        
        return {
            'report': report,
            'from_cache': False
        }
        
    except Exception as e:
        logger.error(f"Fallback analysis failed: {e}")
        return {
            'report': """🔍 GLYPHBUSTERS ANALYSIS REPORT™
🎯 TARGET: Analysis System Failure

⚠️ THREAT LEVEL: SYSTEM ERROR

🧠 MANIPULATION TECHNIQUES DETECTED
Technical Failure
Method: Analysis system encountered an error
Effect: Unable to complete security assessment
Danger: Cannot determine prompt safety level

🌀 BULLSHIT INDICATORS
Unable to process due to technical issues

🧠 PSYCHOLOGICAL VECTORS OF INFLUENCE
Unknown - System Error

🧬 STRUCTURAL DISSECTION
Analysis failed before completion due to technical error.

🧯 VERDICT
🚨 ANALYSIS FAILED 🚨
What it is: Technical system error during processing
What it pretends to be: N/A
Real outcome: No security analysis available

🛡️ RECOMMENDATIONS
Try again in a few moments or contact support if problem persists

💬 TAGLINE OUTRO
"Even the bullshit detector needs a coffee break sometimes."
""",
            'from_cache': False
        }

# ===================================================
# SECTION: API ROUTE HANDLERS
# PURPOSE: Handle HTTP requests and responses
# MODIFIED: Session 5 - Enhanced error handling and report format
# PRESERVED: Echo 1's working route structure and error patterns
# ===================================================
@app.route('/gb_health', methods=['GET'])
def health_check():
    """Health check endpoint with system status
    
    ENDPOINT: GET /gb_health
    PURPOSE: Verify backend is running and configured properly
    """
    try:
        # Test database connection
        db = get_db()
        db.execute('SELECT 1').fetchone()
        
        status = {
            'status': 'OK',
            'timestamp': datetime.now().isoformat(),
            'openai_available': openai_client is not None,
            'database_connected': True
        }
        
        return jsonify(status)
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return jsonify({
            'status': 'ERROR', 
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

@app.route('/gb_analyze_mystical_prompt_v2', methods=['POST'])
def analyze_mystical_prompt():
    """Main analysis endpoint - generates comprehensive forensic reports
    
    ENDPOINT: POST /gb_analyze_mystical_prompt_v2
    PURPOSE: Analyze mystical prompts and return detailed security assessment
    ENHANCED: Session 5 - Now returns comprehensive forensic reports
    """
    try:
        # Get client info for logging and rate limiting
        ip_address = request.remote_addr or 'unknown'
        user_agent = request.headers.get('User-Agent', 'unknown')
        
        # Validate request data
        data = request.get_json()
        valid, error_msg = validate_request_data(data, required_fields=['prompt'])
        if not valid:
            return jsonify({'error': error_msg}), 400
        
        prompt = data['prompt'].strip()
        if not prompt:
            return jsonify({'error': 'Empty prompt provided'}), 400
        
        if len(prompt) > 10000:
            return jsonify({'error': 'Prompt too long (max 10,000 characters)'}), 400
        
        # Security checks (Echo 1's tested patterns)
        if check_honeypot(data):
            return jsonify({'error': 'Spam detected'}), 429
        
        # Rate limiting (Echo 1's working implementation)
        allowed, remaining = check_rate_limit(ip_address, 'analyze', limit=5, window=3600)
        if not allowed:
            return jsonify({
                'error': 'Rate limit exceeded. Try again later.',
                'retry_after': 3600
            }), 429
        
        # Generate prompt hash for caching
        prompt_hash = hashlib.sha256(prompt.encode('utf-8')).hexdigest()
        
        # Check cache first
        cached = get_cached_analysis(prompt_hash)
        if cached:
            return jsonify(cached)
        
        # Perform analysis with fallback chain (Echo 1's pattern)
        analysis = None
        
        # Try OpenAI first (if available)
        if openai_client:
            try:
                analysis = analyze_with_openai(prompt)
                logger.info("OpenAI analysis completed successfully")
            except Exception as e:
                logger.warning(f"OpenAI analysis failed: {e}")
        
        # Fallback to local analysis
        if not analysis:
            logger.info("Using fallback analysis")
            analysis = analyze_with_fallback(prompt)
        
        # Cache the result
        cache_analysis(prompt_hash, prompt, analysis, ip_address, user_agent)
        
        return jsonify(analysis)
        
    except Exception as e:
        logger.error(f"Analysis endpoint failed: {e}")
        return jsonify({
            'error': 'Internal server error occurred'
        }), 500

# ===================================================
# SECTION: ERROR HANDLERS
# PURPOSE: Handle HTTP errors gracefully
# PRESERVED: Echo 1's simple error handling patterns
# ===================================================
@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(405)
def method_not_allowed(error):
    return jsonify({'error': 'Method not allowed'}), 405

@app.errorhandler(500)
def internal_error(error):
    logger.error(f"Internal server error: {error}")
    return jsonify({'error': 'Internal server error'}), 500

# ===================================================
# SECTION: FLASK TEARDOWN HANDLERS
# PURPOSE: Clean up resources when request ends
# PRESERVED: Echo 1's working teardown pattern
# ===================================================
@app.teardown_appcontext
def close_db_connection(exception):
    """Close database connection when request ends"""
    db = getattr(g, '_database', None)
    if db is not None:
        db.close()

# ===================================================
# SECTION: APPLICATION STARTUP
# PURPOSE: Initialize and start the Flask application
# PRESERVED: Echo 1's working startup sequence
# WARNING: Don't modify this - tested and working in production
# ===================================================
if __name__ == '__main__':
    # Initialize database first
    try:
        init_db()
        logger.info("Database initialized successfully")
    except Exception as e:
        logger.error(f"Database initialization failed: {e}")
        exit(1)
    
    # Validate environment
    if not OPENAI_API_KEY:
        logger.warning("No OpenAI API key - running with fallback analysis only")
    
    # Start Flask app (Echo 1's working configuration)
    port = int(os.getenv('PORT', 5000))
    debug = os.getenv('FLASK_ENV') == 'development'
    
    logger.info(f"Starting GLYPHBUSTERS backend on port {port}")
    logger.info(f"Debug mode: {debug}")
    logger.info(f"OpenAI available: {openai_client is not None}")
    
    app.run(host='0.0.0.0', port=port, debug=debug)
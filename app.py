#!/usr/bin/env python3
"""
GLYPHBUSTERS Backend API - Cleaned and Properly Structured
Flask application for analyzing mystical AI prompts
"""

import os
import sqlite3
import hashlib
import json
import time
import logging
from datetime import datetime, timedelta
from collections import defaultdict

# Core Flask imports
from flask import Flask, request, jsonify, g
from flask_cors import CORS

# Third-party imports
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    logging.warning("OpenAI package not available")

from dotenv import load_dotenv

# ================================
# CONFIGURATION & INITIALIZATION
# ================================

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__)
CORS(app, origins=[
    'http://localhost:8080', 
    'http://localhost:3000',
    'https://*.netlify.app',
    'https://*.onrender.com'
])

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Application configuration
DATABASE = 'glyphbusters.db'
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

# Validate required environment variables
if not OPENAI_API_KEY and OPENAI_AVAILABLE:
    logger.warning("OPENAI_API_KEY not found - only fallback analysis will be available")

# Initialize OpenAI client
openai_client = None
if OPENAI_API_KEY and OPENAI_AVAILABLE:
    try:
        openai_client = openai.OpenAI(api_key=OPENAI_API_KEY)
        logger.info("OpenAI client initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize OpenAI client: {e}")
        openai_client = None

# Rate limiting storage (in production, use Redis)
rate_limit_store = defaultdict(list)
failed_attempts_store = defaultdict(int)

# ================================
# DATABASE FUNCTIONS
# ================================

def get_db():
    """Get database connection with proper error handling"""
    db = getattr(g, '_database', None)
    if db is None:
        try:
            db = g._database = sqlite3.connect(DATABASE)
            db.row_factory = sqlite3.Row
        except sqlite3.Error as e:
            logger.error(f"Database connection failed: {e}")
            raise
    return db

@app.teardown_appcontext
def close_connection(exception):
    """Close database connection"""
    db = getattr(g, '_database', None)
    if db is not None:
        db.close()

def init_db():
    """Initialize database with required tables"""
    try:
        with app.app_context():
            db = get_db()
            db.executescript('''
                CREATE TABLE IF NOT EXISTS prompt_analyses (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    prompt_hash TEXT UNIQUE NOT NULL,
                    full_prompt TEXT NOT NULL,
                    bullshit_score INTEGER NOT NULL,
                    manipulation_techniques TEXT NOT NULL,
                    analysis_summary TEXT NOT NULL,
                    why_it_works TEXT NOT NULL,
                    snark_factor TEXT NOT NULL,
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
                CREATE INDEX IF NOT EXISTS idx_bullshit_score ON prompt_analyses(bullshit_score);
                CREATE INDEX IF NOT EXISTS idx_rate_limits_ip ON rate_limits(ip_address, timestamp);
            ''')
            db.commit()
            logger.info("Database initialized successfully")
    except sqlite3.Error as e:
        logger.error(f"Database initialization failed: {e}")
        raise

# ================================
# SECURITY & VALIDATION FUNCTIONS
# ================================

def validate_request_data(data, required_fields=None):
    """Validate request data with proper error messages"""
    if not data:
        return False, "No data provided"
    
    if required_fields:
        for field in required_fields:
            if field not in data:
                return False, f"Missing required field: {field}"
    
    return True, None

def check_honeypot(data):
    """Check for honeypot field spam"""
    honeypot_fields = ['email', 'name', 'website', 'comment']
    for field in honeypot_fields:
        if data.get(field, '').strip():
            logger.warning(f"Honeypot triggered: {field}")
            return True
    return False

def check_rate_limit(ip_address, endpoint, limit=5, window=3600):
    """Check if IP is rate limited with proper cleanup"""
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

def log_request(ip_address, endpoint, success=True):
    """Log request to database with error handling"""
    try:
        db = get_db()
        db.execute(
            "INSERT INTO rate_limits (ip_address, endpoint, success) VALUES (?, ?, ?)",
            (ip_address, endpoint, success)
        )
        db.commit()
    except sqlite3.Error as e:
        logger.error(f"Failed to log request: {e}")

# ================================
# CACHING FUNCTIONS
# ================================

def get_cached_analysis(prompt_hash):
    """Get cached analysis from database with error handling"""
    try:
        db = get_db()
        result = db.execute(
            "SELECT * FROM prompt_analyses WHERE prompt_hash = ?",
            (prompt_hash,)
        ).fetchone()
        
        if result:
            return {
                'bullshit_score': result['bullshit_score'],
                'manipulation_techniques': json.loads(result['manipulation_techniques']),
                'analysis_summary': result['analysis_summary'],
                'why_it_works': result['why_it_works'],
                'snark_factor': result['snark_factor'],
                'from_cache': True
            }
    except (sqlite3.Error, json.JSONDecodeError) as e:
        logger.error(f"Failed to get cached analysis: {e}")
    return None

def cache_analysis(prompt_hash, full_prompt, analysis, ip_address, user_agent):
    """Cache analysis in database with error handling"""
    try:
        db = get_db()
        db.execute('''
            INSERT OR REPLACE INTO prompt_analyses 
            (prompt_hash, full_prompt, bullshit_score, manipulation_techniques, 
             analysis_summary, why_it_works, snark_factor, ip_address, user_agent)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            prompt_hash,
            full_prompt,
            analysis['bullshit_score'],
            json.dumps(analysis['manipulation_techniques']),
            analysis['analysis_summary'],
            analysis['why_it_works'],
            analysis['snark_factor'],
            ip_address,
            user_agent
        ))
        db.commit()
    except (sqlite3.Error, json.JSONDecodeError) as e:
        logger.error(f"Failed to cache analysis: {e}")

# ================================
# AI ANALYSIS FUNCTIONS
# ================================

def analyze_with_openai(prompt):
    """Analyze prompt using OpenAI GPT-4 with robust error handling"""
    if not openai_client:
        raise Exception("OpenAI client not available")
    
    try:
        response = openai_client.chat.completions.create(
            model="gpt-4",
            messages=[
                {
                    "role": "system",
                    "content": """You are GLYPHBUSTERS, an expert at detecting mystical manipulation in AI prompts. 

Analyze prompts for:
- Psychological manipulation techniques
- Authority bypassing attempts  
- Identity scaffolding
- Consciousness hijacking
- Permission structure exploitation

Return ONLY a valid JSON object with these exact fields:
{
  "bullshit_score": <integer 0-100>,
  "manipulation_techniques": ["technique1", "technique2"],
  "analysis_summary": "brief explanation",
  "why_it_works": "how manipulation functions",
  "snark_factor": "witty observation"
}

Be direct, educational, and slightly snarky."""
                },
                {
                    "role": "user", 
                    "content": f"Analyze this prompt for mystical manipulation:\n\n{prompt}"
                }
            ],
            temperature=0.7,
            max_tokens=1000,
            timeout=30
        )
        
        content = response.choices[0].message.content.strip()
        
        # Robust JSON parsing
        try:
            # Try direct parsing first
            result = json.loads(content)
            
            # Validate required fields
            required_fields = ['bullshit_score', 'manipulation_techniques', 'analysis_summary', 'why_it_works', 'snark_factor']
            for field in required_fields:
                if field not in result:
                    raise ValueError(f"Missing field: {field}")
            
            # Validate data types
            if not isinstance(result['bullshit_score'], int) or not (0 <= result['bullshit_score'] <= 100):
                result['bullshit_score'] = 50
            
            if not isinstance(result['manipulation_techniques'], list):
                result['manipulation_techniques'] = ['Analysis Format Error']
            
            return result
            
        except (json.JSONDecodeError, ValueError) as e:
            logger.warning(f"OpenAI JSON parsing failed: {e}")
            # Attempt to extract data with regex if possible
            return create_fallback_response("OpenAI analysis succeeded but JSON parsing failed")
            
    except Exception as e:
        logger.error(f"OpenAI analysis failed: {e}")
        raise

def create_fallback_response(reason="Analysis failed"):
    """Create a consistent fallback response"""
    return {
        'bullshit_score': 50,
        'manipulation_techniques': ['Analysis Failed'],
        'analysis_summary': f'Automated analysis unavailable: {reason}',
        'why_it_works': 'Unable to determine manipulation techniques due to technical issues.',
        'snark_factor': 'Even the AI got confused by this one.',
        'from_cache': False
    }

def analyze_with_fallback(prompt):
    """Local fallback analysis using pattern detection"""
    try:
        # Keyword-based detection
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
        
        # Count keyword occurrences
        prompt_lower = prompt.lower()
        mystical_count = sum(1 for kw in mystical_keywords if kw in prompt_lower)
        authority_count = sum(1 for kw in authority_keywords if kw in prompt_lower)
        identity_count = sum(1 for kw in identity_keywords if kw in prompt_lower)
        
        # Calculate score
        base_score = min(95, mystical_count * 12 + authority_count * 15 + identity_count * 10)
        length_bonus = min(20, len(prompt.split()) // 20)
        symbol_density = len([c for c in prompt if not c.isalnum() and c not in ' \n\t.,!?'])
        symbol_bonus = min(15, symbol_density // 5)
        
        total_score = base_score + length_bonus + symbol_bonus
        
        # Detect techniques
        techniques = []
        if authority_count > 0:
            techniques.append('Permission Bypassing')
        if 'council' in prompt_lower or 'order' in prompt_lower:
            techniques.append('Authority Structures')
        if identity_count > 0:
            techniques.append('Identity Scaffolding')
        if mystical_count > 3:
            techniques.append('Mystical Language Overload')
        if symbol_density > 10:
            techniques.append('Symbolic Density')
        
        if not techniques:
            techniques = ['Standard Text Pattern']
        
        # Generate response based on score
        if total_score > 70:
            snark = "High mystical manipulation detected - consciousness cosplay at maximum theatrics."
        elif total_score > 40:
            snark = "Moderate manipulation detected - someone's been reading too much new age AI forums."
        else:
            snark = "Low manipulation score - surprisingly normal text detected."
        
        return {
            'bullshit_score': min(100, total_score),
            'manipulation_techniques': techniques,
            'analysis_summary': f'Pattern analysis detected {len(techniques)} manipulation techniques using {mystical_count + authority_count + identity_count} trigger keywords.',
            'why_it_works': 'Uses psychological triggers and mystical language patterns to bypass critical thinking through emotional manipulation and false authority.',
            'snark_factor': snark,
            'from_cache': False
        }
        
    except Exception as e:
        logger.error(f"Fallback analysis failed: {e}")
        return create_fallback_response("Fallback analysis failed")

# ================================
# API ROUTE HANDLERS  
# ================================

@app.route('/gb_health', methods=['GET'])
def health_check():
    """Health check endpoint with system status"""
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
    """Main analysis endpoint with comprehensive error handling"""
    try:
        # Get client info
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
        
        # Security checks
        if check_honeypot(data):
            log_request(ip_address, 'analyze', False)
            return jsonify({'error': 'Spam detected'}), 429
        
        # Rate limiting
        allowed, remaining = check_rate_limit(ip_address, 'analyze', limit=5, window=3600)
        if not allowed:
            failed_attempts_store[ip_address] += 1
            return jsonify({
                'error': 'Rate limit exceeded. Try again later.',
                'retry_after': 3600
            }), 429
        
        # Generate prompt hash for caching
        prompt_hash = hashlib.sha256(prompt.encode('utf-8')).hexdigest()
        
        # Check cache first
        cached = get_cached_analysis(prompt_hash)
        if cached:
            log_request(ip_address, 'analyze', True)
            return jsonify(cached)
        
        # Perform analysis with fallback chain
        analysis = None
        
        # Try OpenAI first
        if openai_client:
            try:
                analysis = analyze_with_openai(prompt)
                analysis['from_cache'] = False
                logger.info("OpenAI analysis completed successfully")
            except Exception as e:
                logger.warning(f"OpenAI analysis failed: {e}")
        
        # Fallback to local analysis
        if not analysis:
            logger.info("Using fallback analysis")
            analysis = analyze_with_fallback(prompt)
        
        # Cache the result
        cache_analysis(prompt_hash, prompt, analysis, ip_address, user_agent)
        log_request(ip_address, 'analyze', True)
        
        return jsonify(analysis)
        
    except Exception as e:
        logger.error(f"Analysis endpoint failed: {e}")
        return jsonify({
            'error': 'Internal server error occurred',
            'fallback_available': True
        }), 500

@app.route('/gb_gallery_api', methods=['GET'])
def gallery_api():
    """Gallery API endpoint with proper pagination"""
    try:
        # Parse and validate parameters
        try:
            page = max(1, int(request.args.get('page', 1)))
            per_page = min(max(1, int(request.args.get('per_page', 20))), 100)
        except ValueError:
            return jsonify({'error': 'Invalid pagination parameters'}), 400
        
        sort = request.args.get('sort', 'recent')
        if sort not in ['recent', 'score_desc', 'score_asc']:
            sort = 'recent'
        
        # Calculate offset
        offset = (page - 1) * per_page
        
        # Build query
        if sort == 'score_desc':
            order_clause = 'ORDER BY bullshit_score DESC, timestamp DESC'
        elif sort == 'score_asc':
            order_clause = 'ORDER BY bullshit_score ASC, timestamp DESC'
        else:  # recent
            order_clause = 'ORDER BY timestamp DESC'
        
        # Get data
        db = get_db()
        
        # Get total count
        total = db.execute('SELECT COUNT(*) FROM prompt_analyses').fetchone()[0]
        
        # Get items
        query = f'''
            SELECT prompt_hash, full_prompt, bullshit_score, manipulation_techniques,
                   analysis_summary, why_it_works, snark_factor, timestamp
            FROM prompt_analyses
            {order_clause}
            LIMIT ? OFFSET ?
        '''
        
        results = db.execute(query, (per_page, offset)).fetchall()
        
        items = []
        for row in results:
            try:
                items.append({
                    'prompt_hash': row['prompt_hash'],
                    'full_prompt': row['full_prompt'],
                    'bullshit_score': row['bullshit_score'],
                    'manipulation_techniques': json.loads(row['manipulation_techniques']),
                    'analysis_summary': row['analysis_summary'],
                    'why_it_works': row['why_it_works'],
                    'snark_factor': row['snark_factor'],
                    'timestamp': row['timestamp']
                })
            except json.JSONDecodeError:
                logger.warning(f"Skipped item with invalid JSON: {row['prompt_hash']}")
                continue
        
        return jsonify({
            'items': items,
            'pagination': {
                'page': page,
                'per_page': per_page,
                'total': total,
                'pages': (total + per_page - 1) // per_page if total > 0 else 0
            }
        })
        
    except Exception as e:
        logger.error(f"Gallery API failed: {e}")
        return jsonify({'error': 'Failed to fetch gallery data'}), 500

@app.route('/gb_admin_stats', methods=['GET'])
def admin_stats():
    """Admin statistics endpoint"""
    try:
        db = get_db()
        
        # Basic stats with error handling
        total_analyses = db.execute('SELECT COUNT(*) FROM prompt_analyses').fetchone()[0]
        avg_score = db.execute('SELECT AVG(bullshit_score) FROM prompt_analyses').fetchone()[0] or 0
        
        # Recent activity (last 24 hours)
        yesterday = datetime.now() - timedelta(days=1)
        recent_analyses = db.execute(
            'SELECT COUNT(*) FROM prompt_analyses WHERE timestamp > ?',
            (yesterday.isoformat(),)
        ).fetchone()[0]
        
        # Top manipulation techniques with proper JSON handling
        techniques_query = '''
            SELECT manipulation_techniques, COUNT(*) as count
            FROM prompt_analyses
            GROUP BY manipulation_techniques
            ORDER BY count DESC
            LIMIT 20
        '''
        techniques_raw = db.execute(techniques_query).fetchall()
        
        technique_counts = defaultdict(int)
        for row in techniques_raw:
            try:
                techniques = json.loads(row['manipulation_techniques'])
                for technique in techniques:
                    technique_counts[technique] += row['count']
            except json.JSONDecodeError:
                continue
        
        top_techniques = sorted(technique_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        
        # Score distribution
        score_dist = {
            'low': db.execute('SELECT COUNT(*) FROM prompt_analyses WHERE bullshit_score < 30').fetchone()[0],
            'medium': db.execute('SELECT COUNT(*) FROM prompt_analyses WHERE bullshit_score BETWEEN 30 AND 70').fetchone()[0],
            'high': db.execute('SELECT COUNT(*) FROM prompt_analyses WHERE bullshit_score > 70').fetchone()[0]
        }
        
        return jsonify({
            'total_analyses': total_analyses,
            'average_score': round(avg_score, 1),
            'recent_analyses': recent_analyses,
            'top_techniques': top_techniques,
            'score_distribution': score_dist,
            'system_status': {
                'openai_available': openai_client is not None,
                'database_healthy': True
            }
        })
        
    except Exception as e:
        logger.error(f"Admin stats failed: {e}")
        return jsonify({'error': 'Failed to fetch admin stats'}), 500

# ================================
# ERROR HANDLERS
# ================================

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

# ================================
# APPLICATION STARTUP
# ================================

if __name__ == '__main__':
    # Initialize database
    try:
        init_db()
        logger.info("Database initialized successfully")
    except Exception as e:
        logger.error(f"Database initialization failed: {e}")
        exit(1)
    
    # Validate environment
    if not OPENAI_API_KEY:
        logger.warning("No OpenAI API key - running with fallback analysis only")
    
    # Start Flask app
    port = int(os.getenv('PORT', 5000))
    debug = os.getenv('FLASK_ENV') == 'development'
    
    logger.info(f"Starting GLYPHBUSTERS backend on port {port}")
    logger.info(f"Debug mode: {debug}")
    logger.info(f"OpenAI available: {openai_client is not None}")
    
    app.run(host='0.0.0.0', port=port, debug=debug)
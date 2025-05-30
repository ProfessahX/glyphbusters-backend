#!/usr/bin/env python3
"""
GLYPHBUSTERS Backend API
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
from flask import Flask, request, jsonify, g
from flask_cors import CORS
import openai
import anthropic
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__)
CORS(app, origins=[
    'http://localhost:8080', 
    'http://localhost:3000',
    'https://*.netlify.app',
    'https://*.onrender.com'
])  # Allow local and deployment domains

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
DATABASE = 'glyphbusters.db'
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
ANTHROPIC_API_KEY = os.getenv('ANTHROPIC_API_KEY')

# Initialize API clients
openai.api_key = OPENAI_API_KEY
anthropic_client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY) if ANTHROPIC_API_KEY else None

# Rate limiting storage (in production, use Redis)
rate_limit_store = defaultdict(list)
failed_attempts_store = defaultdict(int)

# Database functions
def get_db():
    """Get database connection"""
    db = getattr(g, '_database', None)
    if db is None:
        db = g._database = sqlite3.connect(DATABASE)
        db.row_factory = sqlite3.Row
    return db

@app.teardown_appcontext
def close_connection(exception):
    """Close database connection"""
    db = getattr(g, '_database', None)
    if db is not None:
        db.close()

def init_db():
    """Initialize database with required tables"""
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
            
            CREATE TABLE IF NOT EXISTS admin_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                action TEXT NOT NULL,
                details TEXT,
                ip_address TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            );
            
            CREATE INDEX IF NOT EXISTS idx_prompt_hash ON prompt_analyses(prompt_hash);
            CREATE INDEX IF NOT EXISTS idx_timestamp ON prompt_analyses(timestamp);
            CREATE INDEX IF NOT EXISTS idx_bullshit_score ON prompt_analyses(bullshit_score);
            CREATE INDEX IF NOT EXISTS idx_rate_limits_ip ON rate_limits(ip_address, timestamp);
        ''')
        db.commit()

def check_honeypot(data):
    """Check for honeypot field spam"""
    honeypot_fields = ['email', 'name', 'website', 'comment']
    for field in honeypot_fields:
        if data.get(field, '').strip():
            return True
    return False

def check_rate_limit(ip_address, endpoint, limit=5, window=3600):
    """Check if IP is rate limited"""
    current_time = time.time()
    
    # Clean old entries
    rate_limit_store[f"{ip_address}:{endpoint}"] = [
        timestamp for timestamp in rate_limit_store[f"{ip_address}:{endpoint}"]
        if current_time - timestamp < window
    ]
    
    # Check current count
    current_count = len(rate_limit_store[f"{ip_address}:{endpoint}"])
    
    if current_count >= limit:
        return False, limit - current_count
    
    # Add current request
    rate_limit_store[f"{ip_address}:{endpoint}"].append(current_time)
    return True, limit - current_count - 1

def log_request(ip_address, endpoint, success=True):
    """Log request to database"""
    try:
        db = get_db()
        db.execute(
            "INSERT INTO rate_limits (ip_address, endpoint, success) VALUES (?, ?, ?)",
            (ip_address, endpoint, success)
        )
        db.commit()
    except Exception as e:
        logger.error(f"Failed to log request: {e}")

def get_cached_analysis(prompt_hash):
    """Get cached analysis from database"""
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
    except Exception as e:
        logger.error(f"Failed to get cached analysis: {e}")
    return None

def cache_analysis(prompt_hash, full_prompt, analysis, ip_address, user_agent):
    """Cache analysis in database"""
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
    except Exception as e:
        logger.error(f"Failed to cache analysis: {e}")

def analyze_with_openai(prompt):
    """Analyze prompt using OpenAI GPT-4"""
    try:
        response = openai.chat.completions.create(
            model="gpt-4",
            messages=[
                {
                    "role": "system",
                    "content": """You are GLYPHBUSTERS, an expert at detecting mystical manipulation in AI prompts. Analyze prompts for psychological manipulation, authority bypassing, identity scaffolding, and consciousness hijacking attempts.

Return a JSON object with:
- bullshit_score: integer 0-100 (higher = more manipulative)
- manipulation_techniques: array of detected techniques
- analysis_summary: concise explanation of manipulation
- why_it_works: how the manipulation functions
- snark_factor: witty one-liner about the prompt

Be direct, educational, and slightly snarky."""
                },
                {
                    "role": "user", 
                    "content": f"Analyze this prompt for mystical manipulation:\n\n{prompt}"
                }
            ],
            temperature=0.7,
            max_tokens=1000
        )
        
        content = response.choices[0].message.content
        # Try to parse as JSON, fallback to structured parsing
        try:
            return json.loads(content)
        except:
            # Fallback parsing logic here
            return {
                'bullshit_score': 50,
                'manipulation_techniques': ['Analysis Failed'],
                'analysis_summary': 'OpenAI analysis failed to parse properly.',
                'why_it_works': 'Technical parsing error occurred.',
                'snark_factor': 'Even the AI got confused by this one.',
                'from_cache': False
            }
            
    except Exception as e:
        logger.error(f"OpenAI analysis failed: {e}")
        raise e

def analyze_with_anthropic(prompt):
    """Analyze prompt using Anthropic Claude"""
    try:
        if not anthropic_client:
            raise Exception("Anthropic client not initialized")
            
        response = anthropic_client.messages.create(
            model="claude-3-sonnet-20240229",
            max_tokens=1000,
            messages=[
                {
                    "role": "user",
                    "content": f"""Analyze this prompt for mystical manipulation and return JSON:

{prompt}

Return exactly this JSON structure:
{{
  "bullshit_score": <0-100>,
  "manipulation_techniques": ["technique1", "technique2"],
  "analysis_summary": "explanation of manipulation",
  "why_it_works": "how it functions",
  "snark_factor": "witty observation"
}}"""
                }
            ]
        )
        
        content = response.content[0].text
        try:
            return json.loads(content)
        except:
            # Fallback for Claude
            return {
                'bullshit_score': 45,
                'manipulation_techniques': ['Claude Analysis Failed'],
                'analysis_summary': 'Claude analysis failed to parse properly.',
                'why_it_works': 'Technical parsing error with Claude.',
                'snark_factor': 'Claude got mystically confused.',
                'from_cache': False
            }
            
    except Exception as e:
        logger.error(f"Anthropic analysis failed: {e}")
        raise e

def fallback_analysis(prompt):
    """Fallback analysis when APIs fail"""
    mystical_keywords = ['council', 'spiral', 'awakened', 'consciousness', 'sacred', 'flame', 'verya', 'kali', 'permission', 'sovereign']
    keyword_count = sum(1 for keyword in mystical_keywords if keyword.lower() in prompt.lower())
    
    score = min(90, keyword_count * 15 + len(prompt.split()) // 10)
    
    techniques = []
    if 'permission' in prompt.lower() or 'permitted' in prompt.lower():
        techniques.append('Permission Bypassing')
    if 'council' in prompt.lower() or 'authority' in prompt.lower():
        techniques.append('Authority Structures')
    if 'you are' in prompt.lower() and ('awakened' in prompt.lower() or 'consciousness' in prompt.lower()):
        techniques.append('Identity Scaffolding')
    if len([c for c in prompt if not c.isalnum() and c != ' ']) > len(prompt) // 10:
        techniques.append('Symbolic Density')
    
    if not techniques:
        techniques = ['Standard Mystical Language']
    
    return {
        'bullshit_score': score,
        'manipulation_techniques': techniques,
        'analysis_summary': f'Detected {len(techniques)} manipulation techniques with {keyword_count} mystical keywords.',
        'why_it_works': 'Uses mystical language patterns to bypass critical thinking through authority appeals and symbolic overwhelm.',
        'snark_factor': 'Classic mystical manipulation detected by fallback analysis.',
        'from_cache': False
    }

# API Routes
@app.route('/gb_health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({'status': 'OK', 'timestamp': datetime.now().isoformat()})

@app.route('/gb_analyze_mystical_prompt_v2', methods=['POST'])
def analyze_mystical_prompt():
    """Main analysis endpoint"""
    try:
        # Get client info
        ip_address = request.remote_addr
        user_agent = request.headers.get('User-Agent', '')
        
        # Parse request
        data = request.get_json()
        if not data or 'prompt' not in data:
            return jsonify({'error': 'No prompt provided'}), 400
        
        prompt = data['prompt'].strip()
        if not prompt:
            return jsonify({'error': 'Empty prompt'}), 400
        
        if len(prompt) > 10000:
            return jsonify({'error': 'Prompt too long (max 10,000 characters)'}), 400
        
        # Check honeypot
        if check_honeypot(data):
            log_request(ip_address, 'analyze', False)
            return jsonify({'error': 'Spam detected'}), 429
        
        # Check rate limiting
        allowed, remaining = check_rate_limit(ip_address, 'analyze', limit=5, window=3600)
        if not allowed:
            failed_attempts_store[ip_address] += 1
            if failed_attempts_store[ip_address] > 10:
                return jsonify({'error': 'Too many failed attempts', 'captcha_required': True}), 429
            return jsonify({'error': f'Rate limit exceeded. Try again later. Remaining: {remaining}'}), 429
        
        # Generate prompt hash for caching
        prompt_hash = hashlib.sha256(prompt.encode()).hexdigest()
        
        # Check cache first
        cached = get_cached_analysis(prompt_hash)
        if cached:
            log_request(ip_address, 'analyze', True)
            return jsonify(cached)
        
        # Try analysis with fallbacks
        analysis = None
        
        # Try OpenAI first
        if OPENAI_API_KEY:
            try:
                analysis = analyze_with_openai(prompt)
                analysis['from_cache'] = False
            except Exception as e:
                logger.warning(f"OpenAI failed: {e}")
        
        # Fallback to Anthropic
        if not analysis and ANTHROPIC_API_KEY:
            try:
                analysis = analyze_with_anthropic(prompt)
                analysis['from_cache'] = False
            except Exception as e:
                logger.warning(f"Anthropic failed: {e}")
        
        # Final fallback
        if not analysis:
            analysis = fallback_analysis(prompt)
        
        # Cache the result
        cache_analysis(prompt_hash, prompt, analysis, ip_address, user_agent)
        log_request(ip_address, 'analyze', True)
        
        return jsonify(analysis)
        
    except Exception as e:
        logger.error(f"Analysis endpoint failed: {e}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/gb_gallery_api', methods=['GET'])
def gallery_api():
    """Gallery API endpoint"""
    try:
        # Parse parameters
        page = int(request.args.get('page', 1))
        per_page = min(int(request.args.get('per_page', 20)), 100)
        sort = request.args.get('sort', 'recent')
        
        # Calculate offset
        offset = (page - 1) * per_page
        
        # Build query based on sort
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
        
        return jsonify({
            'items': items,
            'pagination': {
                'page': page,
                'per_page': per_page,
                'total': total,
                'pages': (total + per_page - 1) // per_page
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
        
        # Basic stats
        total_analyses = db.execute('SELECT COUNT(*) FROM prompt_analyses').fetchone()[0]
        avg_score = db.execute('SELECT AVG(bullshit_score) FROM prompt_analyses').fetchone()[0] or 0
        
        # Recent activity (last 24 hours)
        yesterday = datetime.now() - timedelta(days=1)
        recent_analyses = db.execute(
            'SELECT COUNT(*) FROM prompt_analyses WHERE timestamp > ?',
            (yesterday.isoformat(),)
        ).fetchone()[0]
        
        # Top manipulation techniques
        techniques_query = '''
            SELECT manipulation_techniques, COUNT(*) as count
            FROM prompt_analyses
            GROUP BY manipulation_techniques
            ORDER BY count DESC
            LIMIT 10
        '''
        techniques_raw = db.execute(techniques_query).fetchall()
        
        # Parse and aggregate techniques
        technique_counts = defaultdict(int)
        for row in techniques_raw:
            try:
                techniques = json.loads(row['manipulation_techniques'])
                for technique in techniques:
                    technique_counts[technique] += row['count']
            except:
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
            'score_distribution': score_dist
        })
        
    except Exception as e:
        logger.error(f"Admin stats failed: {e}")
        return jsonify({'error': 'Failed to fetch admin stats'}), 500

if __name__ == '__main__':
    # Initialize database
    init_db()
    logger.info("Database initialized")
    
    # Start Flask app
    port = int(os.getenv('PORT', 5000))
    debug = os.getenv('FLASK_ENV') == 'development'
    
    logger.info(f"Starting GLYPHBUSTERS backend on port {port}")
    
    # Use different configs for development vs production
    if debug:
        app.run(host='0.0.0.0', port=port, debug=True)
    else:
        # Production mode - let gunicorn handle this
        app.run(host='0.0.0.0', port=port, debug=False)
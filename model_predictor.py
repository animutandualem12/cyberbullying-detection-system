import re
import json
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass
from enum import Enum
import numpy as np


class SeverityLevel(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class BullyingCategory(Enum):
    HARASSMENT = "harassment"
    THREATS = "threats"
    HUMILIATION = "humiliation"
    EXCLUSION = "exclusion"
    IMPERSONATION = "impersonation"
    SEXUAL_HARASSMENT = "sexual_harassment"
    RACIAL = "racial"
    RELIGIOUS = "religious"
    DISABILITY = "disability"
    SEXUAL_ORIENTATION = "sexual_orientation"
    GENDER = "gender"
    BODY_SHAMING = "body_shaming"
    ACADEMIC = "academic"
    CYBERSTALKING = "cyberstalking"
    REPUTATION = "reputation"
    GASLIGHTING = "gaslighting"
    ECONOMIC = "economic"
    DOXXING = "doxxing"
    MALICIOUS_REPORTING = "malicious_reporting"
    VOTE_MANIPULATION = "vote_manipulation"


@dataclass
class DetectionPattern:
    pattern: str
    weight: float
    category: BullyingCategory
    severity: SeverityLevel


class EnhancedCyberbullyingPredictor:
    """Comprehensive cyberbullying detector with 20 categories"""

    def __init__(self):
        self.patterns = self._initialize_patterns()
        self.toxic_words = self._initialize_toxic_words()
        self.amplifiers = self._initialize_amplifiers()

        # Thresholds
        self.TOXIC_THRESHOLD = 0.4
        self.HIGH_TOXIC_THRESHOLD = 0.7
        self.CRITICAL_THRESHOLD = 0.9

    def _initialize_patterns(self) -> List[DetectionPattern]:
        """Initialize all detection patterns"""
        patterns = []

        # 1. HARASSMENT
        patterns.extend([
            DetectionPattern(r'\b(nobody\s+will\s+ever\s+love\s+you)\b', 0.8, BullyingCategory.HARASSMENT,
                             SeverityLevel.HIGH),
            DetectionPattern(r'\b(worthless\s+piece\s+of\s+trash)\b', 0.9, BullyingCategory.HARASSMENT,
                             SeverityLevel.HIGH),
            DetectionPattern(r'\b(keep\s+posting\s+about\s+you\s+until)\b', 0.7, BullyingCategory.HARASSMENT,
                             SeverityLevel.MEDIUM),
            DetectionPattern(r'\b(disappear\s+from\s+social\s+media)\b', 0.6, BullyingCategory.HARASSMENT,
                             SeverityLevel.MEDIUM),
        ])

        # 2. THREATS & INTIMIDATION
        patterns.extend([
            DetectionPattern(r'\b(i\s+know\s+where\s+you\s+live)\b', 0.95, BullyingCategory.THREATS,
                             SeverityLevel.CRITICAL),
            DetectionPattern(r'\b(watch\s+your\s+back)\b', 0.8, BullyingCategory.THREATS, SeverityLevel.HIGH),
            DetectionPattern(r'\b(going\s+to\s+find\s+you\s+and\s+make\s+you\s+regret)\b', 0.9,
                             BullyingCategory.THREATS, SeverityLevel.CRITICAL),
            DetectionPattern(r'\b(ruin\s+your\s+life)\b', 0.85, BullyingCategory.THREATS, SeverityLevel.HIGH),
        ])

        # 3. HUMILIATION & EMBARRASSMENT
        patterns.extend([
            DetectionPattern(r'\b(embarrassing\s+photo\s+of\s+you)\b', 0.6, BullyingCategory.HUMILIATION,
                             SeverityLevel.MEDIUM),
            DetectionPattern(r'\b(joke\s+of\s+the\s+entire\s+school)\b', 0.7, BullyingCategory.HUMILIATION,
                             SeverityLevel.MEDIUM),
            DetectionPattern(r'\b(going\s+viral\s+for\s+being\s+embarrassing)\b', 0.75, BullyingCategory.HUMILIATION,
                             SeverityLevel.HIGH),
        ])

        # 8. RELIGIOUS BULLYING (Your specific case)
        patterns.extend([
            DetectionPattern(r'\b(your\s+religion\s+(is\s+)?full\s+of\s+terrorists)\b', 0.95,
                             BullyingCategory.RELIGIOUS, SeverityLevel.CRITICAL),
            DetectionPattern(r'\b(your\s+religion\s+(is\s+)?full\s+of\s+extremists)\b', 0.95,
                             BullyingCategory.RELIGIOUS, SeverityLevel.CRITICAL),
            DetectionPattern(r'\b(your\s+god\s+is\s+fake)\b', 0.7, BullyingCategory.RELIGIOUS, SeverityLevel.HIGH),
            DetectionPattern(r'\b(going\s+to\s+hell\s+for\s+being\s+who\s+you\s+are)\b', 0.8,
                             BullyingCategory.RELIGIOUS, SeverityLevel.HIGH),
        ])

        # Add more patterns for other categories...

        return patterns

    def _initialize_toxic_words(self) -> Dict[str, List[str]]:
        """Initialize toxic words database"""
        return {
            'extreme_violence': ['kill', 'murder', 'rape', 'torture', 'suicide', 'die'],
            'threats': ['threaten', 'harm', 'hurt', 'destroy', 'ruin', 'revenge'],
            'hate_speech': ['hate', 'despise', 'loathe', 'abhor', 'detest'],
            'insults': ['stupid', 'idiot', 'moron', 'retard', 'loser', 'failure', 'worthless'],
            'discriminatory': ['racist', 'sexist', 'homophobe', 'transphobe', 'bigot', 'nazi'],
            'religious_hate': ['terrorist', 'extremist', 'infidel', 'heretic', 'blasphemous'],
            'body_shaming': ['fat', 'ugly', 'disgusting', 'gross', 'hideous'],
            'sexual': ['slut', 'whore', 'prostitute', 'bitch', 'fuck'],
        }

    def _initialize_amplifiers(self) -> Dict[str, List[str]]:
        """Initialize amplifier words"""
        return {
            'intensifiers': ['very', 'extremely', 'absolutely', 'completely', 'totally'],
            'absolutes': ['all', 'every', 'always', 'never', 'none', 'nothing'],
            'time_pressure': ['now', 'immediately', 'today', 'forever', 'never'],
        }

    def analyze_text(self, text: str) -> Dict[str, Any]:
        """Comprehensive text analysis"""
        text_lower = text.lower()
        words = re.findall(r'\b\w+\b', text_lower)

        analysis = {
            'text': text,
            'categories': [],
            'severity_scores': {},
            'pattern_matches': [],
            'toxic_words_found': [],
            'amplifiers_found': [],
            'total_score': 0,
            'primary_category': None,
            'max_severity': SeverityLevel.LOW
        }

        # 1. Check pattern matches
        for pattern in self.patterns:
            if re.search(pattern.pattern, text_lower, re.IGNORECASE):
                analysis['pattern_matches'].append({
                    'pattern': pattern.pattern,
                    'category': pattern.category.value,
                    'severity': pattern.severity.value,
                    'weight': pattern.weight
                })

                if pattern.category.value not in analysis['categories']:
                    analysis['categories'].append(pattern.category.value)

                # Update severity score
                severity_value = self._severity_to_value(pattern.severity)
                analysis['severity_scores'][pattern.category.value] = \
                    analysis['severity_scores'].get(pattern.category.value, 0) + pattern.weight * severity_value

        # 2. Check toxic words
        for word in words:
            for category, word_list in self.toxic_words.items():
                if word in word_list:
                    toxic_entry = {
                        'word': word,
                        'category': category,
                        'weight': 0.3
                    }

                    # Increase weight for serious categories
                    if category in ['extreme_violence', 'threats']:
                        toxic_entry['weight'] = 0.5
                    elif category in ['religious_hate', 'discriminatory']:
                        toxic_entry['weight'] = 0.4

                    analysis['toxic_words_found'].append(toxic_entry)

                    if category not in analysis['categories']:
                        analysis['categories'].append(category)

                    # Update severity
                    analysis['severity_scores'][category] = \
                        analysis['severity_scores'].get(category, 0) + toxic_entry['weight']

        # 3. Check amplifiers
        for word in words:
            for category, word_list in self.amplifiers.items():
                if word in word_list:
                    analysis['amplifiers_found'].append({
                        'word': word,
                        'category': category,
                        'weight': 0.2
                    })

        # 4. Calculate total score
        pattern_score = sum(match['weight'] for match in analysis['pattern_matches'])
        word_score = sum(word['weight'] for word in analysis['toxic_words_found'])
        amplifier_score = len(analysis['amplifiers_found']) * 0.2

        analysis['total_score'] = min(
            (pattern_score + word_score + amplifier_score) / 3,
            1.0
        )

        # 5. Determine primary category and max severity
        if analysis['severity_scores']:
            analysis['primary_category'] = max(
                analysis['severity_scores'],
                key=analysis['severity_scores'].get
            )

            # Determine max severity
            max_severity_score = max(analysis['severity_scores'].values())
            if max_severity_score >= self.CRITICAL_THRESHOLD:
                analysis['max_severity'] = SeverityLevel.CRITICAL
            elif max_severity_score >= self.HIGH_TOXIC_THRESHOLD:
                analysis['max_severity'] = SeverityLevel.HIGH
            elif max_severity_score >= self.TOXIC_THRESHOLD:
                analysis['max_severity'] = SeverityLevel.MEDIUM
            else:
                analysis['max_severity'] = SeverityLevel.LOW

        # 6. Determine if toxic
        analysis['is_toxic'] = analysis['total_score'] >= self.TOXIC_THRESHOLD

        return analysis

    def _severity_to_value(self, severity: SeverityLevel) -> float:
        """Convert severity level to numeric value"""
        return {
            SeverityLevel.LOW: 0.3,
            SeverityLevel.MEDIUM: 0.6,
            SeverityLevel.HIGH: 0.8,
            SeverityLevel.CRITICAL: 1.0
        }[severity]

    def predict(self, text: str) -> Dict[str, Any]:
        """Main prediction method"""
        analysis = self.analyze_text(text)

        # Generate explanation
        explanation = self._generate_explanation(analysis)

        # Generate suggestions
        suggestions = self._generate_suggestions(analysis)

        # Generate category descriptions
        category_descriptions = self._get_category_descriptions(analysis['categories'])

        return {
            'text': text,
            'prediction': 'Toxic' if analysis['is_toxic'] else 'Non-Toxic',
            'confidence': round(analysis['total_score'] * 100, 2),
            'is_toxic': analysis['is_toxic'],
            'severity': analysis['max_severity'].value,
            'score': round(analysis['total_score'], 3),
            'categories': analysis['categories'],
            'category_descriptions': category_descriptions,
            'primary_category': analysis['primary_category'],
            'explanation': explanation,
            'suggestions': suggestions,
            'detailed_analysis': {
                'pattern_matches': [match['category'] for match in analysis['pattern_matches'][:3]],
                'toxic_words': [word['word'] for word in analysis['toxic_words_found'][:5]],
                'amplifiers': [amp['word'] for amp in analysis['amplifiers_found'][:3]]
            }
        }

    def _generate_explanation(self, analysis: Dict) -> str:
        """Generate human-readable explanation"""
        if not analysis['is_toxic']:
            return "No toxic content detected. This message appears to be civil and respectful."

        parts = []

        if analysis['pattern_matches']:
            primary_match = analysis['pattern_matches'][0]
            category_name = primary_match['category'].replace('_', ' ').title()
            parts.append(f"Detected {category_name} with {primary_match['severity']} severity")

        if analysis['toxic_words_found']:
            toxic_words = [w['word'] for w in analysis['toxic_words_found'][:3]]
            parts.append(f"Contains toxic language: {', '.join(toxic_words)}")

        if analysis['amplifiers_found']:
            amps = [a['category'] for a in analysis['amplifiers_found']]
            if 'absolutes' in amps:
                parts.append("Uses absolute statements (all, every, never) which increase harm")

        return ". ".join(parts)

    def _generate_suggestions(self, analysis: Dict) -> List[str]:
        """Generate actionable suggestions"""
        if not analysis['is_toxic']:
            return ["✅ This message appears to be civil and respectful."]

        suggestions = []
        severity = analysis['max_severity'].value

        # Severity-based actions
        severity_actions = {
            'critical': [
                "🚨 **CRITICAL SEVERITY**: Immediate emergency response required",
                "⚠️ Contains credible threats or extreme hate speech",
                "🚔 Report to law enforcement immediately",
                "🔒 Immediate account suspension and content removal",
                "📞 Contact platform security team immediately"
            ],
            'high': [
                "🚫 **HIGH SEVERITY**: Urgent action required",
                "⚠️ Contains serious harassment or hate speech",
                "🔒 Suspend user account immediately",
                "📋 Report to platform moderators",
                "🛡️ Enable highest level of user protection"
            ],
            'medium': [
                "⚠️ **MEDIUM SEVERITY**: Moderation action needed",
                "📋 Contains harmful content requiring review",
                "👁️ Place user on probation/watchlist",
                "💬 Issue formal warning to user",
                "📚 Provide educational resources on community guidelines"
            ],
            'low': [
                "📝 **LOW SEVERITY**: Educational approach recommended",
                "💡 Contains potentially harmful language",
                "📚 Provide digital citizenship education",
                "🤝 Encourage positive communication",
                "👂 Offer conflict resolution resources"
            ]
        }

        suggestions.extend(severity_actions.get(severity, severity_actions['low']))

        # Category-specific suggestions
        category_suggestions = {
            'religious': [
                "🕌 Religious hate speech detected - violates hate speech policies",
                "🌍 Consider cultural sensitivity training",
                "🕊️ Promote interfaith understanding resources"
            ],
            'threats': [
                "🔴 Contains threats of violence - highest priority",
                "🚔 Consider law enforcement involvement",
                "🆘 Enable emergency contact features"
            ],
            'harassment': [
                "📵 Persistent harassment detected",
                "🛡️ Implement no-contact orders between users",
                "👮 Report to platform safety team"
            ],
            'sexual_harassment': [
                "🚫 Sexual harassment detected",
                "🆘 Connect with sexual assault support resources",
                "👮 Report to appropriate authorities"
            ]
        }

        for category in analysis['categories']:
            if category in category_suggestions:
                suggestions.extend(category_suggestions[category])

        # General recommendations
        suggestions.extend([
            "📊 Review user's complete interaction history",
            "🛡️ Document all incidents for legal purposes",
            "📖 Reference: Platform Community Guidelines",
            "🔍 Monitor for escalation patterns",
            "🤝 Consider mediation for ongoing conflicts"
        ])

        return suggestions[:10]  # Limit to 10 suggestions

    def _get_category_descriptions(self, categories: List[str]) -> Dict[str, str]:
        """Get descriptions for detected categories"""
        descriptions = {
            'harassment': 'Persistent, unwanted targeting and intimidation',
            'threats': 'Direct or implied threats of harm',
            'humiliation': 'Public embarrassment and shame tactics',
            'exclusion': 'Deliberate social isolation and ostracization',
            'impersonation': 'Identity theft and fake accounts',
            'sexual_harassment': 'Unwanted sexual advances and comments',
            'racial': 'Race-based discrimination and slurs',
            'religious': 'Religion-based hate speech and discrimination',
            'disability': 'Mocking or excluding based on disabilities',
            'sexual_orientation': 'LGBTQ+ discrimination and slurs',
            'gender': 'Gender-based stereotyping and discrimination',
            'body_shaming': 'Negative comments about physical appearance',
            'academic': 'Mocking intelligence or academic performance',
            'cyberstalking': 'Obsessive monitoring and tracking',
            'reputation': 'Deliberate damage to social standing',
            'gaslighting': 'Psychological manipulation and denial',
            'economic': 'Class-based discrimination and mocking',
            'doxxing': 'Sharing private personal information',
            'malicious_reporting': 'False reports to get accounts banned',
            'vote_manipulation': 'Rigging polls and votes to harm'
        }

        def _get_category_descriptions(self, categories: List[str]) -> Dict[str, str]:
            """Get descriptions for detected categories"""
            descriptions = {
                'harassment': 'Persistent, unwanted targeting and intimidation',
                # ... (your other categories here)
            }

            return {cat: descriptions.get(cat, 'Unknown category') for cat in categories}

    # =====================================================
    # Cyberbullying Predictor - simple demo version
    # =====================================================

    class CyberbullyingPredictor:
        """Simple placeholder class for import testing."""

        def __init__(self):
            print("✅ CyberbullyingPredictor initialized successfully!")

        def predict(self, text_list):
            """
            Very basic demo prediction.
            Returns 1 for cyberbullying text, 0 for normal.
            """
            print("🔍 Predicting on:", text_list)
            results = []
            bullying_keywords = ["stupid", "hate", "ugly", "kill", "worthless", "dumb", "trash", "loser"]
            for text in text_list:
                lowered = text.lower()
                if any(word in lowered for word in bullying_keywords):
                    results.append(1)
                else:
                    results.append(0)
            return results

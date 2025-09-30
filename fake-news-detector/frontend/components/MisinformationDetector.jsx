import React, { useState } from 'react';
import { Search, AlertTriangle, CheckCircle, XCircle, HelpCircle, Globe, FileText, Shield, Download, ThumbsUp, ThumbsDown } from 'lucide-react';

const MisinformationDetectorApp = () => {
  const [currentStep, setCurrentStep] = useState('input');
  const [inputType, setInputType] = useState('url');
  const [inputValue, setInputValue] = useState('');
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [analysisResults, setAnalysisResults] = useState(null);
  const [selectedClaim, setSelectedClaim] = useState(null);
  const [apiUrl] = useState('http://localhost:8000');

  const analyzeContent = async () => {
    if (!inputValue.trim()) {
      alert('Please enter content to analyze');
      return;
    }

    setIsAnalyzing(true);
    setCurrentStep('analyzing');
    
    try {
      const requestBody = inputType === 'url' 
        ? { url: inputValue, include_evidence: true, include_propagation: true }
        : { text: inputValue, include_evidence: true };

      const response = await fetch(`${apiUrl}/analyze`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(requestBody)
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const result = await response.json();
      setAnalysisResults(result);
      setCurrentStep('results');
      
    } catch (error) {
      console.error('Analysis failed:', error);
      alert('Analysis failed. Please check if the backend is running at ' + apiUrl);
      setCurrentStep('input');
    } finally {
      setIsAnalyzing(false);
    }
  };

  const submitFeedback = async (rating) => {
    if (!analysisResults) return;

    try {
      await fetch(`${apiUrl}/feedback`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          analysis_id: 'temp_id',
          rating: rating === 'accurate' ? 5 : rating === 'mostly' ? 4 : rating === 'mixed' ? 3 : 2
        })
      });
      
      alert('Thank you for your feedback!');
    } catch (error) {
      console.error('Feedback error:', error);
    }
  };

  const getVerdictIcon = (verdict) => {
    switch (verdict) {
      case 'supported': return <CheckCircle className="w-5 h-5 text-green-600" />;
      case 'contradicted': return <XCircle className="w-5 h-5 text-red-600" />;
      case 'ambiguous': return <HelpCircle className="w-5 h-5 text-yellow-600" />;
      default: return <HelpCircle className="w-5 h-5 text-gray-600" />;
    }
  };

  const getScoreColor = (score) => {
    if (score >= 70) return 'text-green-600 bg-green-50 border-green-200';
    if (score >= 40) return 'text-yellow-600 bg-yellow-50 border-yellow-200';
    return 'text-red-600 bg-red-50 border-red-200';
  };

  const InputScreen = () => (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 via-white to-indigo-50">
      <div className="container mx-auto px-4 py-12">
        <div className="text-center mb-12">
          <div className="inline-flex items-center justify-center w-16 h-16 bg-blue-600 rounded-2xl mb-6">
            <Shield className="w-8 h-8 text-white" />
          </div>
          <h1 className="text-4xl font-bold text-gray-900 mb-4">VERITAS</h1>
          <p className="text-xl text-gray-600 max-w-2xl mx-auto">
            Advanced AI-powered fact-checking and misinformation detection platform
          </p>
        </div>

        <div className="max-w-4xl mx-auto">
          <div className="bg-white rounded-3xl shadow-xl border border-gray-100 overflow-hidden">
            <div className="p-8">
              <div className="text-center mb-8">
                <h2 className="text-2xl font-semibold text-gray-900 mb-2">VERIFY THE TRUTH</h2>
                <p className="text-gray-600">Paste a URL or text to analyze for misinformation</p>
              </div>

              <div className="flex mb-6 bg-gray-50 rounded-xl p-1">
                <button
                  onClick={() => setInputType('url')}
                  className={`flex-1 py-3 px-6 rounded-lg font-medium transition-all ${
                    inputType === 'url'
                      ? 'bg-white text-blue-600 shadow-sm'
                      : 'text-gray-600 hover:text-gray-900'
                  }`}
                >
                  <Globe className="w-4 h-4 inline mr-2" />
                  URL
                </button>
                <button
                  onClick={() => setInputType('text')}
                  className={`flex-1 py-3 px-6 rounded-lg font-medium transition-all ${
                    inputType === 'text'
                      ? 'bg-white text-blue-600 shadow-sm'
                      : 'text-gray-600 hover:text-gray-900'
                  }`}
                >
                  <FileText className="w-4 h-4 inline mr-2" />
                  Text
                </button>
              </div>

              <div className="mb-6">
                {inputType === 'url' ? (
                  <input
                    type="url"
                    value={inputValue}
                    onChange={(e) => setInputValue(e.target.value)}
                    placeholder="https://example.com/news-article"
                    className="w-full px-4 py-4 text-lg border-2 border-gray-200 rounded-xl focus:border-blue-500 focus:outline-none transition-colors"
                  />
                ) : (
                  <textarea
                    value={inputValue}
                    onChange={(e) => setInputValue(e.target.value)}
                    placeholder="Paste article text, social media post, or any content to analyze..."
                    rows={6}
                    className="w-full px-4 py-4 text-lg border-2 border-gray-200 rounded-xl focus:border-blue-500 focus:outline-none transition-colors resize-none"
                  />
                )}
              </div>

              <button
                onClick={analyzeContent}
                disabled={!inputValue.trim()}
                className="w-full bg-blue-600 text-white py-4 px-8 rounded-xl font-semibold text-lg hover:bg-blue-700 disabled:bg-gray-300 disabled:cursor-not-allowed transition-all transform hover:scale-[1.02] disabled:hover:scale-100"
              >
                <Search className="w-5 h-5 inline mr-2" />
                Analyze Content
              </button>

              <div className="mt-6 text-center">
                <button
                  onClick={() => {
                    setInputValue("Scientists have discovered a revolutionary new battery technology that can store 10 times more energy than current lithium-ion batteries. This breakthrough will solve climate change overnight by making renewable energy storage incredibly cheap and efficient.");
                    setInputType('text');
                  }}
                  className="text-blue-600 hover:text-blue-700 font-medium"
                >
                  Try with sample text
                </button>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );

  const AnalyzingScreen = () => (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 via-white to-indigo-50 flex items-center justify-center">
      <div className="text-center">
        <div className="w-24 h-24 bg-blue-600 rounded-full flex items-center justify-center mx-auto mb-8 animate-pulse">
          <Search className="w-12 h-12 text-white animate-spin" />
        </div>
        <h2 className="text-3xl font-bold text-gray-900 mb-4">Analyzing Content</h2>
        <p className="text-gray-600 mb-8 max-w-md mx-auto">
          Our AI is examining the content for misinformation patterns, checking claims against verified sources, and analyzing credibility indicators.
        </p>
        <div className="flex justify-center space-x-8 text-sm text-gray-500">
          <div className="flex items-center">
            <div className="w-2 h-2 bg-blue-600 rounded-full animate-pulse mr-2"></div>
            Extracting claims
          </div>
          <div className="flex items-center">
            <div className="w-2 h-2 bg-blue-600 rounded-full animate-pulse mr-2" style={{animationDelay: '0.5s'}}></div>
            Verifying sources
          </div>
          <div className="flex items-center">
            <div className="w-2 h-2 bg-blue-600 rounded-full animate-pulse mr-2" style={{animationDelay: '1s'}}></div>
            Analyzing credibility
          </div>
        </div>
      </div>
    </div>
  );

  const ResultsScreen = () => {
    if (!analysisResults) return null;

    const supportedCount = analysisResults.claims.filter(c => c.verdict === 'supported').length;
    const contradictedCount = analysisResults.claims.filter(c => c.verdict === 'contradicted').length;
    const ambiguousCount = analysisResults.claims.filter(c => c.verdict === 'ambiguous').length;

    return (
      <div className="min-h-screen bg-gray-50">
        <div className="bg-white border-b border-gray-200 px-6 py-4">
          <div className="max-w-6xl mx-auto flex items-center justify-between">
            <div className="flex items-center space-x-4">
              <button 
                onClick={() => setCurrentStep('input')}
                className="flex items-center text-blue-600 hover:text-blue-700 font-medium"
              >
                ← Back to Analysis
              </button>
              <div className="w-px h-6 bg-gray-300"></div>
              <div className="flex items-center space-x-2">
                <Shield className="w-5 h-5 text-blue-600" />
                <span className="font-semibold text-gray-900">VERITAS</span>
              </div>
            </div>
            <button className="flex items-center px-4 py-2 text-gray-600 hover:text-gray-900">
              <Download className="w-4 h-4 mr-2" />
              Export Report
            </button>
          </div>
        </div>

        <div className="max-w-6xl mx-auto px-6 py-8">
          <div className="bg-white rounded-2xl shadow-sm border border-gray-200 p-8 mb-8">
            <div className="flex items-center justify-between mb-6">
              <div>
                <h1 className="text-3xl font-bold text-gray-900 mb-2">ANALYSIS RESULTS</h1>
                <p className="text-gray-600">Content credibility assessment complete</p>
              </div>
              <div className={`text-center p-6 rounded-2xl border-2 ${getScoreColor(analysisResults.overall_credibility_score)}`}>
                <div className="text-4xl font-bold mb-1">{analysisResults.overall_credibility_score}%</div>
                <div className="text-sm font-medium">CREDIBILITY SCORE</div>
              </div>
            </div>

            <div className="grid grid-cols-2 md:grid-cols-4 gap-6">
              <div className="text-center">
                <div className="text-2xl font-bold text-gray-900 mb-1">{analysisResults.claims.length}</div>
                <div className="text-sm text-gray-600">Claims Detected</div>
              </div>
              <div className="text-center">
                <div className="text-2xl font-bold text-red-600 mb-1">{contradictedCount}</div>
                <div className="text-sm text-gray-600">Contradicted</div>
              </div>
              <div className="text-center">
                <div className="text-2xl font-bold text-yellow-600 mb-1">{ambiguousCount}</div>
                <div className="text-sm text-gray-600">Needs Verification</div>
              </div>
              <div className="text-center">
                <div className="text-2xl font-bold text-green-600 mb-1">{supportedCount}</div>
                <div className="text-sm text-gray-600">Supported</div>
              </div>
            </div>

            <div className="mt-6 p-4 bg-red-50 border-l-4 border-red-400 rounded-r-lg">
              <div className="flex items-center">
                <AlertTriangle className="w-5 h-5 text-red-600 mr-2" />
                <span className="font-semibold text-red-800">{analysisResults.risk_level} of Misinformation</span>
              </div>
            </div>
          </div>

          <div className="grid lg:grid-cols-3 gap-8">
            <div className="lg:col-span-2 space-y-6">
              <div className="bg-white rounded-2xl shadow-sm border border-gray-200 p-6">
                <h2 className="text-xl font-bold text-gray-900 mb-6">Detailed Claims Analysis</h2>
                
                <div className="space-y-4">
                  {analysisResults.claims.map((claim, index) => (
                    <div 
                      key={index}
                      className={`p-4 rounded-xl border-2 cursor-pointer transition-all hover:shadow-md ${
                        claim.verdict === 'contradicted' ? 'border-red-200 bg-red-50' : 
                        claim.verdict === 'supported' ? 'border-green-200 bg-green-50' :
                        'border-yellow-200 bg-yellow-50'
                      } ${selectedClaim === index ? 'ring-2 ring-blue-500' : ''}`}
                      onClick={() => setSelectedClaim(selectedClaim === index ? null : index)}
                    >
                      <div className="flex items-start justify-between mb-3">
                        <div className="flex items-center space-x-2">
                          {getVerdictIcon(claim.verdict)}
                          <span className="font-semibold text-sm uppercase tracking-wide">
                            {claim.verdict}
                          </span>
                        </div>
                        <div className="text-sm text-gray-600">
                          {Math.round(claim.confidence * 100)}% confidence
                        </div>
                      </div>
                      
                      <p className="text-gray-900 mb-3 font-medium">"{claim.text}"</p>
                      
                      {selectedClaim === index && claim.evidence && claim.evidence.length > 0 && (
                        <div className="mt-4 pt-4 border-t border-gray-200">
                          <h4 className="font-semibold text-gray-900 mb-3">Supporting Evidence:</h4>
                          <div className="space-y-3">
                            {claim.evidence.map((evidence, idx) => (
                              <div key={idx} className="bg-white p-3 rounded-lg border border-gray-200">
                                <div className="flex items-center justify-between mb-2">
                                  <span className="font-medium text-blue-600 text-sm">{evidence.source}</span>
                                  <span className="text-xs text-gray-500">{evidence.credibility}% reliable</span>
                                </div>
                                <p className="text-sm text-gray-700">{evidence.text}</p>
                              </div>
                            ))}
                          </div>
                        </div>
                      )}
                    </div>
                  ))}
                </div>
              </div>
            </div>

            <div className="space-y-6">
              <div className="bg-white rounded-2xl shadow-sm border border-gray-200 p-6">
                <h3 className="text-lg font-bold text-gray-900 mb-4">Source Analysis</h3>
                <div className="space-y-4">
                  <div>
                    <div className="flex justify-between items-center mb-2">
                      <span className="text-sm font-medium text-gray-700">Domain Credibility</span>
                      <span className="text-sm font-bold">{Math.round(analysisResults.source_credibility.credibility_score * 100)}%</span>
                    </div>
                    <div className="w-full bg-gray-200 rounded-full h-2">
                      <div 
                        className={`h-2 rounded-full ${
                          analysisResults.source_credibility.credibility_score > 0.7 ? 'bg-green-500' :
                          analysisResults.source_credibility.credibility_score > 0.4 ? 'bg-yellow-500' :
                          'bg-red-500'
                        }`}
                        style={{ width: `${analysisResults.source_credibility.credibility_score * 100}%` }}
                      ></div>
                    </div>
                  </div>
                  
                  <div className="grid grid-cols-2 gap-3 text-sm">
                    <div>
                      <span className="text-gray-600">Domain:</span>
                      <div className="font-medium">{analysisResults.source_credibility.domain}</div>
                    </div>
                    <div>
                      <span className="text-gray-600">Reputation:</span>
                      <div className="font-medium">{analysisResults.source_credibility.reputation}</div>
                    </div>
                  </div>
                </div>
              </div>

              <div className="bg-white rounded-2xl shadow-sm border border-gray-200 p-6">
                <h3 className="text-lg font-bold text-gray-900 mb-4">Content Patterns</h3>
                <div className="space-y-3">
                  <div className="flex justify-between items-center">
                    <span className="text-sm text-gray-700">Sensational Language</span>
                    <span className="text-sm font-bold text-red-600">{analysisResults.content_analysis.sensational_words_count}</span>
                  </div>
                  <div className="flex justify-between items-center">
                    <span className="text-sm text-gray-700">Hedging Language</span>
                    <span className="text-sm font-bold text-green-600">{analysisResults.content_analysis.hedging_words_count}</span>
                  </div>
                  <div className="flex justify-between items-center">
                    <span className="text-sm text-gray-700">Clickbait Score</span>
                    <span className="text-sm font-bold text-yellow-600">{Math.round(analysisResults.content_analysis.clickbait_score)}%</span>
                  </div>
                </div>
              </div>

              <div className="bg-white rounded-2xl shadow-sm border border-gray-200 p-6">
                <h3 className="text-lg font-bold text-gray-900 mb-4">Help Us Improve</h3>
                <p className="text-sm text-gray-600 mb-4">Was this analysis helpful?</p>
                <div className="flex space-x-3">
                  <button 
                    onClick={() => submitFeedback('accurate')}
                    className="flex-1 flex items-center justify-center py-2 px-3 bg-green-50 text-green-700 rounded-lg hover:bg-green-100 transition-colors"
                  >
                    <ThumbsUp className="w-4 h-4 mr-1" />
                    Yes
                  </button>
                  <button 
                    onClick={() => submitFeedback('inaccurate')}
                    className="flex-1 flex items-center justify-center py-2 px-3 bg-red-50 text-red-700 rounded-lg hover:bg-red-100 transition-colors"
                  >
                    <ThumbsDown className="w-4 h-4 mr-1" />
                    No
                  </button>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    );
  };

  return (
    <div className="min-h-screen">
      {currentStep === 'input' && <InputScreen />}
      {currentStep === 'analyzing' && <AnalyzingScreen />}
      {currentStep === 'results' && <ResultsScreen />}
    </div>
  );
};

export default MisinformationDetectorApp;
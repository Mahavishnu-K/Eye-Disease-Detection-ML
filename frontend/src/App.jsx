import { useState, useEffect, useRef } from 'react'
import axios from 'axios'
import './App.css'
import { BsSend } from "react-icons/bs";
import ChatService from './services/ChatService'

function App() {
  const [file, setFile] = useState(null)
  const [preview, setPreview] = useState(null)
  const [result, setResult] = useState(null)
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState(null)
  
  const [showChat, setShowChat] = useState(false)
  const [messages, setMessages] = useState([])
  const [input, setInput] = useState('')
  const [chatLoading, setChatLoading] = useState(false)
  const messagesEndRef = useRef(null)

  const handleFileChange = (e) => {
    const selectedFile = e.target.files[0]
    if (selectedFile) {
      setFile(selectedFile)
      setPreview(URL.createObjectURL(selectedFile))
      setResult(null)
      setError(null)
    }
  }

  const handleSubmit = async (e) => {
    e.preventDefault()
    if (!file) return

    setIsLoading(true)
    setError(null)

    const formData = new FormData()
    formData.append('file', file)

    try {
      const response = await axios.post('/api/predict', formData, {
        headers: {
          'Content-Type': 'multipart/form-data'
        }
      })
      setResult(response.data)
    } catch (err) {
      setError(err.response?.data?.error || 'An error occurred')
    } finally {
      setIsLoading(false)
    }
  }

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }

  useEffect(() => {
    scrollToBottom()
  }, [messages])

  const sendMessage = async (e) => {
    e.preventDefault()
    if (!input.trim()) return

    const userMessage = {
      role: 'user',
      content: input
    }

    setMessages([...messages, userMessage])
    setInput('')
    setChatLoading(true)

    try {
      const response = await ChatService.sendMessage([...messages, userMessage])

      const assistantMessage = {
        role: 'system',
        content: response.choices[0].message.content
      }

      setMessages(prev => [...prev, assistantMessage])
    } catch (err) {
      console.error('Chat error:', err)
      setMessages(prev => [...prev, {
        role: 'system',
        content: 'Sorry, I encountered an error. Please try again.'
      }])
    } finally {
      setChatLoading(false)
    }
  }

  const toggleChat = () => {
    setShowChat(!showChat)
    if (!messages.length) {
      setMessages([{
        role: 'system',
        content: 'Hello! I can help answer questions about eye diseases and the diagnostic process. How can I assist you today?'
      }])
    }
  }

  const formatSystemMessage = (content) => {
    let formattedContent = content;
    
    formattedContent = formattedContent.replace(/---/g, '<hr/>');
    
    formattedContent = formattedContent.replace(/### (.*?)$/gm, '<h3>$1</h3>');
    
    formattedContent = formattedContent.replace(/#### (.*?)$/gm, '<h4>$1</h4>');

    formattedContent = formattedContent.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
    
    formattedContent = formattedContent.replace(/- (.*?)$/gm, '<li>$1</li>');
    
    formattedContent = formattedContent.replace(/<li>(.*?)<\/li>(\s*)<li>/g, '<li>$1</li><li>');
    formattedContent = formattedContent.replace(/<li>(.*?)(\s*)<\/li>/g, '<ul><li>$1</li></ul>');
    
    formattedContent = formattedContent.replace(/<\/ul>\s*<ul>/g, '');
    
    const paragraphs = formattedContent.split('\n\n');
    formattedContent = paragraphs.map(p => {
      if (!p.trim()) return '';
      if (p.includes('<h3>') || p.includes('<h4>') || p.includes('<ul>') || p.includes('<hr/>')) {
        return p;
      }
      return `<p>${p}</p>`;
    }).join('');
    
    return <div dangerouslySetInnerHTML={{ __html: formattedContent }} />;
  };

  return (
    <div className="app-container">
      <header className="app-header">
        <h1>Eye Disease Detection</h1>
        <p>Upload an eye image to detect potential diseases</p>
      </header>

      <main className="app-main">
        <form onSubmit={handleSubmit} className="upload-form">
          <div className="file-input-container">
            <input
              id="file-upload"
              type="file"
              onChange={handleFileChange}
              accept="image/*"
              className="file-input"
            />
            <label htmlFor="file-upload" className="file-label">
              {file ? file.name : 'Choose an image...'}
            </label>
          </div>
          
          {preview && (
            <div className="image-preview">
              <img src={preview} alt="Preview" />
            </div>
          )}

          <button
            type="submit"
            disabled={!file || isLoading}
            className="submit-button"
          >
            {isLoading ? 'Analyzing...' : 'Analyze Image'}
          </button>
        </form>

        {error && <div className="error-message">{error}</div>}

        {result && (
          <div className="result-container">
            <div className="prediction-result">
              <h2>Diagnosis: {result.prediction}</h2>
              <p className="confidence">Confidence: {result.confidence}</p>
            </div>

            <div className="image-results">
              <div className="image-result">
                <h3>Original Image</h3>
                <img 
                  src={`data:image/png;base64,${result.original_image}`} 
                  alt="Original" 
                />
              </div>
              <div className="image-result">
                <h3>Grad-CAM Heatmap</h3>
                <img 
                  src={`data:image/png;base64,${result.grad_cam_image}`} 
                  alt="Grad-CAM Visualization" 
                />
              </div>
            </div>

            {/* Add button to ask about results */}
            <button 
              className="ask-about-results-button"
              onClick={() => {
                setShowChat(true);
                if (!messages.some(m => m.content.includes("diagnosis"))) {
                  setMessages(prev => [
                    ...prev, 
                    {
                      role: 'system',
                      content: `The system has detected ${result.prediction} with ${result.confidence} confidence. Would you like to know more about this condition?`
                    }
                  ]);
                }
              }}
            >
              Ask about this diagnosis
            </button>
          </div>
        )}
      </main>

      {/* Chat toggle button */}
      <button 
        className="chat-toggle-button"
        onClick={toggleChat}
      >
        Ask AI Assistant
      </button>

      {/* Chat window with overlay */}
      {showChat && (
        <div className="chat-overlay">
          <div className="chat-container">
            <div className="chat-header">
              <h3>Chat with Eye Care AI Assistant</h3>
              <button className="close-chat" onClick={toggleChat}>✕</button>
            </div>
            
            <div className="messages-container">
              {messages.map((msg, index) => (
                <div 
                  key={index} 
                  className={`message ${msg.role === 'user' ? 'user-message' : 'system-message'}`}
                >
                    {msg.role !== 'user' && (
                      <div className="message-role">
                        AI Assistant
                      </div>
                    )}
                  {msg.role === 'system' 
                    ? formatSystemMessage(msg.content) 
                    : msg.content
                  }
                </div>
              ))}
              {chatLoading && <div className="loading-indicator">...</div>}
              <div ref={messagesEndRef} />
            </div>
            
            <form onSubmit={sendMessage} className="chat-input-form">
              <input
                type="text"
                value={input}
                onChange={(e) => setInput(e.target.value)}
                placeholder="Ask about eye health or diagnosis..."
                disabled={chatLoading}
              />
              <button type="submit" disabled={chatLoading}>
                <BsSend size={24}/>
              </button>
            </form>
          </div>
        </div>
      )}
    </div>
  )
}

export default App
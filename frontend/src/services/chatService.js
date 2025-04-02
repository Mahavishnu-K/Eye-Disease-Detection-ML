import OpenAI from "openai";

class ChatService {
  constructor() {
    this.openai = new OpenAI({
      apiKey: import.meta.env.VITE_OPENAI_API_KEY,
      baseURL: import.meta.env.VITE_OPENAI_BASE_URL,
      dangerouslyAllowBrowser: true 
    });
  }

  async sendMessage(messages) {
    try {
      const formattedMessages = [
        {
          role: "system",
          content: "You are an AI assistant specialized in eye diseases. Provide helpful, accurate information about eye conditions, diagnostics, and treatments. When appropriate, remind users to seek professional medical advice."
        },
        ...messages.map(msg => ({
          role: msg.role === 'system' ? 'assistant' : msg.role,
          content: msg.content
        }))
      ];

      const response = await this.openai.chat.completions.create({
        model: "gpt-4o",
        messages: formattedMessages,
        temperature: 1,
        max_tokens: 4096,
        top_p: 1,
      });

      return response;
    } catch (error) {
      console.error('Error with OpenAI API:', error);
      throw new Error('Failed to communicate with the AI service');
    }
  }
}

export default new ChatService();
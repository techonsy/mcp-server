#!/usr/bin/env node

import { Server } from '@modelcontextprotocol/sdk/server/index.js';
import { StdioServerTransport } from '@modelcontextprotocol/sdk/server/stdio.js';
import {
  CallToolRequestSchema,
  ListToolsRequestSchema,
  Tool,
} from '@modelcontextprotocol/sdk/types.js';
import { createClient } from '@supabase/supabase-js';
import { CohereClient } from "cohere-ai";
import { OpenAI } from "openai";
import { v4 as uuidv4 } from 'uuid';
import { google } from 'googleapis';
import { OAuth2Client } from 'google-auth-library';

// Initialize Supabase client
const supabase = createClient(
    process.env.NEXT_PUBLIC_SUPABASE_URL || '',
   process.env.SUPABASE_SERVICE_ROLE_KEY || ''
);

// Initialize Cohere client
const cohere = new CohereClient({
  token: process.env.COHERE_API_KEY,
});

// Initialize OpenAI client
const openai = new OpenAI({
  baseURL:"https://openrouter.ai/api/v1",
  apiKey:process.env.OPENROUTER_API_KEY,
});

// Google Calendar configuration
const GOOGLE_CLIENT_ID = process.env.GOOGLE_CLIENT_ID || '';
const GOOGLE_CLIENT_SECRET = process.env.GOOGLE_CLIENT_SECRET || '';
const GOOGLE_REDIRECT_URI = process.env.GOOGLE_REDIRECT_URI || 'http://localhost:3000/auth/callback';

// Keep your existing interfaces
interface Message {
    id: string;
    content: string;
    created_at: string;
    embedding?: number[];
    sender_id: string;  
    receiver_id: string;
    is_task_created?: boolean;
    is_system?: boolean;
}

interface TaskExtraction {
  task: string;
  priority: 'low' | 'medium' | 'high' | 'urgent';
  confidence: number;
  description: string;
  due_date: string | null;
  start_date: string | null;
  start_time: string | null;
  end_time: string | null;
  action: 'create' | 'update' | 'complete' | 'cancel';
  existing_task_reference?: string;
  matched_task_id?: string; 
  update_fields?: string[];
}

interface Task {
  id?: string;
  content: string;
  priority: 'low' | 'medium' | 'high' | 'urgent';
  confidence: number;
  description: string;
  message_id?: string;
  sender_id: string;
  receiver_id: string;
  status: 'pending' | 'completed' | 'cancelled';
  created_at: string;
  due_date?: string | null;
  start_date?: string | null;
  start_time?: string | null;
  end_time?: string | null;
  completed_at?: string | null;
  embedding?: number[];
  calendar_event_id?: string | null;
}

interface TaskSimilarity {
  task: Task;
  similarity: number;
  reasons: string[];
}

interface UserCalendarAuth {
  user_id: string;
  access_token: string;
  refresh_token: string;
  expires_at: string;
}

// Google Calendar helper functions
async function getGoogleCalendarClient(userId: string): Promise<any> {
  try {
    const { data: authData, error } = await supabase
      .from('user_google_tokens')
      .select('*')
      .eq('user_id', userId)
      .single();

    if (error || !authData) {
      throw new Error(`No Google authentication found for user ${userId}`);
    }

    const oauth2Client = new OAuth2Client(
      GOOGLE_CLIENT_ID,
      GOOGLE_CLIENT_SECRET,
      GOOGLE_REDIRECT_URI
    );

    const now = Date.now();
    const expiresAt = Number(authData.expiry_date);
    
    if (now >= expiresAt) {
      oauth2Client.setCredentials({
        refresh_token: authData.refresh_token
      });
      
      const { credentials } = await oauth2Client.refreshAccessToken();
      
      await supabase
        .from('user_google_tokens')
        .update({
          access_token: credentials.access_token,
          refresh_token: credentials.refresh_token || authData.refresh_token,
          expiry_date: credentials.expiry_date?.toString() || (now + 3600000).toString(),
          updated_at: new Date().toISOString()
        })
        .eq('user_id', userId);
      
      oauth2Client.setCredentials(credentials);
    } else {
      oauth2Client.setCredentials({
        access_token: authData.access_token,
        refresh_token: authData.refresh_token
      });
    }

    return google.calendar({ version: 'v3', auth: oauth2Client });
  } catch (error) {
    console.error('Error getting Google Calendar client:', error);
    throw error;
  }
}

async function checkCalendarEligibility(userId: string): Promise<{
  eligible: boolean;
  reason: string;
  has_auth: boolean;
  token_valid: boolean;
  calendar_access: boolean;
  expiry_date?: string;
}> {
  try {
    const { data: authData, error } = await supabase
      .from('user_google_tokens')
      .select('expiry_date')
      .eq('user_id', userId)
      .single();

    if (error || !authData) {
      return {
        eligible: false,
        reason: 'No Google authentication found',
        has_auth: false,
        token_valid: false,
        calendar_access: false
      };
    }

    const now = Date.now();
    const expiresAt = Number(authData.expiry_date);
    const tokenValid = now < expiresAt;

    let calendarAccess = false;
    if (tokenValid) {
      try {
        const oauth2Client = new OAuth2Client(
          GOOGLE_CLIENT_ID,
          GOOGLE_CLIENT_SECRET,
          GOOGLE_REDIRECT_URI
        );
        
        calendarAccess = true;
      } catch (error) {
        console.error('Calendar access verification failed:', error);
      }
    }

    return {
      eligible: tokenValid && calendarAccess,
      reason: tokenValid ? 
        (calendarAccess ? 'Full access available' : 'No calendar access') : 
        'Token expired',
      has_auth: true,
      token_valid: tokenValid,
      calendar_access: calendarAccess,
      expiry_date: new Date(expiresAt).toISOString()
    };
  } catch (error) {
    console.error('Error checking calendar eligibility:', error);
    return {
      eligible: false,
      reason: 'Error checking eligibility',
      has_auth: false,
      token_valid: false,
      calendar_access: false
    };
  }
}

async function createCalendarEvent(calendar: any, task: Task): Promise<string | null> {
  try {
    const startDateTime = task.start_date && task.start_time 
      ? new Date(`${task.start_date}T${task.start_time}`)
      : task.due_date 
        ? new Date(task.due_date)
        : new Date();

    const endDateTime = task.start_date && task.end_time
      ? new Date(`${task.start_date}T${task.end_time}`)
      : new Date(startDateTime.getTime() + 60 * 60 * 1000); // Default 1 hour duration

    const event = {
      summary: task.content,
      description: `${task.description}\n\nPriority: ${task.priority}\nTask ID: ${task.id}`,
      start: {
        dateTime: startDateTime.toISOString(),
        timeZone: 'UTC',
      },
      end: {
        dateTime: endDateTime.toISOString(),
        timeZone: 'UTC',
      },
      reminders: {
        useDefault: false,
        overrides: [
          { method: 'email', minutes: 24 * 60 }, // 1 day before
          { method: 'popup', minutes: 30 }, // 30 minutes before
        ],
      },
    };

    const response = await calendar.events.insert({
      calendarId: 'primary',
      requestBody: event,
    });

    return response.data.id;
  } catch (error) {
    console.error('Error creating calendar event:', error);
    return null;
  }
}

async function updateCalendarEvent(calendar: any, eventId: string, task: Task): Promise<boolean> {
  try {
    const startDateTime = task.start_date && task.start_time 
      ? new Date(`${task.start_date}T${task.start_time}`)
      : task.due_date 
        ? new Date(task.due_date)
        : new Date();

    const endDateTime = task.start_date && task.end_time
      ? new Date(`${task.start_date}T${task.end_time}`)
      : new Date(startDateTime.getTime() + 60 * 60 * 1000);

    const event = {
      summary: task.content,
      description: `${task.description}\n\nPriority: ${task.priority}\nTask ID: ${task.id}`,
      start: {
        dateTime: startDateTime.toISOString(),
        timeZone: 'UTC',
      },
      end: {
        dateTime: endDateTime.toISOString(),
        timeZone: 'UTC',
      },
    };

    await calendar.events.update({
      calendarId: 'primary',
      eventId: eventId,
      requestBody: event,
    });

    return true;
  } catch (error) {
    console.error('Error updating calendar event:', error);
    return false;
  }
}

async function deleteCalendarEvent(calendar: any, eventId: string): Promise<boolean> {
  try {
    await calendar.events.delete({
      calendarId: 'primary',
      eventId: eventId,
    });
    return true;
  } catch (error) {
    console.error('Error deleting calendar event:', error);
    return false;
  }
}

// Keep your existing utility functions
async function generateEmbedding(text: string): Promise<number[]> {
  try {
    const embed = await cohere.embed({
      texts: [text],
      model: "embed-english-v3.0",
      inputType: "search_document",
    });

    if (
      !embed?.embeddings ||
      !Array.isArray(embed.embeddings) ||
      embed.embeddings.length === 0 ||
      !Array.isArray(embed.embeddings[0]) ||
      embed.embeddings[0].length === 0
    ) {
      throw new Error("Failed to generate embedding");
    }

    return embed.embeddings[0];
  } catch (err) {
    console.error("Embedding error:", err);
    return [];
  }
}

function cosineSimilarity(a: number[], b: number[]): number {
  if (a.length !== b.length) return 0;
  
  let dotProduct = 0;
  let normA = 0;
  let normB = 0;
  
  for (let i = 0; i < a.length; i++) {
    dotProduct += a[i] * b[i];
    normA += a[i] * a[i];
    normB += b[i] * b[i];
  }
  
  return dotProduct / (Math.sqrt(normA) * Math.sqrt(normB));
}

async function findRecentTasks(senderId: string, receiverId: string, limit: number = 20): Promise<Task[]> {
  try {
    const { data: tasks, error } = await supabase
      .from('tasks')
      .select('*, embedding')
      .or(`and(sender_id.eq.${senderId},receiver_id.eq.${receiverId}),and(sender_id.eq.${receiverId},receiver_id.eq.${senderId})`)
      .in('status', ['pending', 'completed']) 
      .order('created_at', { ascending: false })
      .limit(limit);

    if (error) {
      console.error('Error fetching recent tasks:', error);
      return [];
    }
    return tasks || [];
  } catch (error) {
    console.error('Error finding recent tasks:', error);
    return [];
  }
}

// Enhanced processTaskMessage with calendar integration
async function processTaskMessage(content: string, senderId: string, receiverId: string): Promise<any> {
  try {
    // Generate embedding for the message
    const embedding = await generateEmbedding(content);
    
    // Get recent tasks for context (with embeddings)
    const recentTasks = await findRecentTasks(senderId, receiverId);
    
    // Create a temporary message object for processing
    const tempMessage: Message = {
      id: uuidv4(),
      content,
      created_at: new Date().toISOString(),
      embedding,
      sender_id: senderId,
      receiver_id: receiverId,
      is_system: true
    };

    // Find relevant tasks for potential updates
    const relevantTasks: TaskSimilarity[] = [];
    for (const task of recentTasks.filter(t => t.status === 'pending')) {
      const taskEmbedding = task.embedding && Array.isArray(task.embedding) ? task.embedding : [];
      if (taskEmbedding.length === 0) continue;
      const similarity = cosineSimilarity(embedding, taskEmbedding);
      
      if (similarity > 0.5) {
        const reasons = [];
        if (similarity > 0.7) reasons.push(`High semantic similarity (${(similarity * 100).toFixed(1)}%)`);
        
        const messageWords = content.toLowerCase().split(/\s+/);
        const taskWords = (task.content + ' ' + task.description).toLowerCase().split(/\s+/);
        const commonWords = messageWords.filter(word => 
          word.length > 3 && taskWords.some(taskWord => taskWord.includes(word) || word.includes(taskWord))
        );
        if (commonWords.length > 0) reasons.push(`Common keywords: ${commonWords.join(', ')}`);
        
        relevantTasks.push({ task, similarity, reasons });
      }
    }

    // Extract task using AI (enhanced with calendar fields)
    const taskExtraction = await extractTaskFromMessage(content, recentTasks, relevantTasks);
    
    let messageInserted = false;
    if (taskExtraction && taskExtraction.action === 'create') {
      const { error: messageError } = await supabase
        .from('messages')
        .insert([{
          id: tempMessage.id,
          content: tempMessage.content,
          sender_id: tempMessage.sender_id,
          receiver_id: tempMessage.receiver_id,
          created_at: tempMessage.created_at,
          embedding: tempMessage.embedding,
          is_system: true
        }]);
      if (messageError) {
        throw messageError;
      }
      messageInserted = true;
    }

    if (taskExtraction) {
      const result = await executeTaskAction(taskExtraction, senderId, receiverId, tempMessage.id);
      return {
        success: true,
        action: taskExtraction.action,
        task: taskExtraction.task,
        confidence: taskExtraction.confidence,
        result,
        messageInserted
      };
    }
    
    return {
      success: false,
      message: "No task action detected"
    };
    
  } catch (error) {
    console.error('Error processing task message:', error);
    return {
      success: false,
      error: error instanceof Error ? error.message : 'Unknown error occurred'
    };
  }
}

// Enhanced extractTaskFromMessage with calendar fields
async function extractTaskFromMessage(
  currentMessage: string,
  recentTasks: Task[] = [],
  relevantTasks: TaskSimilarity[] = []
): Promise<TaskExtraction | null> {
  try {
    const tasksContext = recentTasks.length > 0
      ? recentTasks.map((task, i) => 
          `Task ${i + 1} (ID: ${task.id}): "${task.content}" - Priority: ${task.priority}, Status: ${task.status}, Due: ${task.due_date || 'No deadline'}, Start: ${task.start_date || 'No start date'}`
        ).join('\n')
      : "No recent tasks found.";

    const relevantTasksContext = relevantTasks.length > 0
      ? relevantTasks.map((item, i) => 
          `Relevant Task ${i + 1} (ID: ${item.task.id}, Similarity: ${(item.similarity * 100).toFixed(1)}%):\n           Content: "${item.task.content}"\n           Priority: ${item.task.priority}, Status: ${item.task.status}\n           Reasons: ${item.reasons.join(', ')}`
        ).join('\n\n')
      : "No relevant tasks found.";

    const currentDate = new Date().toISOString().split('T')[0];

    const completion = await openai.chat.completions.create({
      model: "meta-llama/llama-3.3-8b-instruct:free",
      messages: [
        {
          role: "system",
          content: `You are a task management AI with calendar integration. Analyze messages to determine: CREATE, UPDATE, COMPLETE, or CANCEL tasks with calendar events.
          
CURRENT DATE: ${currentDate}

RESPONSE FORMAT (JSON only):
{
  "task": "task content",
  "priority": "low|medium|high|urgent",
  "confidence": 0.0-1.0,
  "description": "action description",
  "due_date": "YYYY-MM-DD or null",
  "start_date": "YYYY-MM-DD or null",
  "start_time": "HH:MM or null",
  "end_time": "HH:MM or null",
  "action": "create|update|complete|cancel",
  "matched_task_id": "task ID if updating/completing/canceling",
  "update_fields": ["content", "priority", "due_date", "start_date", "start_time", "end_time", "status"]
}

Extract dates and times carefully. Look for:
- Meeting times: "at 3pm", "from 2-4pm", "9:30 AM"
- Date references: "tomorrow", "next Monday", "December 15th"
- Duration hints: "1 hour meeting", "30 minute call"`
        },
        {
          role: "user",
          content: `MESSAGE: "${currentMessage}"

RECENT TASKS:
${tasksContext}

RELEVANT TASKS:
${relevantTasksContext}

Analyze and respond with JSON only.`
        }
      ],
      temperature: 0.2,
      max_tokens: 800
    });

    const response = completion.choices[0]?.message?.content;
    if (!response) return null;

    const parsed = JSON.parse(response.trim()) as TaskExtraction;
    
    if (parsed.action === 'create' && (!parsed.task || parsed.task === 'null')) {
      return null;
    }

    if (['update', 'complete', 'cancel'].includes(parsed.action) && !parsed.matched_task_id && relevantTasks.length > 0) {
      parsed.matched_task_id = relevantTasks[0].task.id;
    }

    return parsed;
  } catch (error) {
    console.error('Error extracting task:', error);
    return null;
  }
}

// Enhanced executeTaskAction with calendar integration
async function executeTaskAction(extraction: TaskExtraction, senderId: string, receiverId: string, messageId: string): Promise<any> {
  try {
    switch (extraction.action) {
      case 'create':
        // Generate embedding for the new task
        const newTaskText = [extraction.task, extraction.description].filter(Boolean).join(' ');
        const newTaskEmbedding = await generateEmbedding(newTaskText);

        const dueDate = extraction.due_date === 'null' ? null : extraction.due_date;
        const startDate = extraction.start_date === 'null' ? null : extraction.start_date;
        const startTime = extraction.start_time === 'null' ? null : extraction.start_time;
        const endTime = extraction.end_time === 'null' ? null : extraction.end_time;

        const newTask: Task = {
          content: extraction.task,
          priority: extraction.priority,
          confidence: extraction.confidence,
          description: extraction.description,
          message_id: messageId,
          sender_id: senderId,
          receiver_id: receiverId,
          status: 'pending',
          created_at: new Date().toISOString(),
          due_date: dueDate,
          start_date: startDate,
          start_time: startTime,
          end_time: endTime,
          embedding: newTaskEmbedding
        };

        const { data: createdTask, error: createError } = await supabase
          .from('tasks')
          .insert([newTask])
          .select()
          .single();

        if (createError) throw createError;

        // Create calendar event
        let calendarEventId = null;
        try {
          const calendar = await getGoogleCalendarClient(senderId);
          calendarEventId = await createCalendarEvent(calendar, createdTask);

          if (calendarEventId) {
            await supabase
              .from('tasks')
              .update({ calendar_event_id: calendarEventId })
              .eq('id', createdTask.id);
          }
        } catch (calendarError) {
          console.error('Calendar event creation failed:', calendarError);
        }

        return { created: { ...createdTask, calendar_event_id: calendarEventId } };

      case 'update':
        if (!extraction.matched_task_id) throw new Error('No task ID for update');

        const updates: Partial<Task> = {};
        let shouldUpdateEmbedding = false;

        if (extraction.update_fields?.includes('content')) {
          updates.content = extraction.task;
          shouldUpdateEmbedding = true;
        }
        if (extraction.update_fields?.includes('description')) {
          updates.description = extraction.description;
          shouldUpdateEmbedding = true;
        }
        if (extraction.update_fields?.includes('priority')) updates.priority = extraction.priority;
        if (extraction.update_fields?.includes('due_date')) {
          updates.due_date = extraction.due_date === 'null' ? null : extraction.due_date;
        }
        if (extraction.update_fields?.includes('start_date')) {
          updates.start_date = extraction.start_date === 'null' ? null : extraction.start_date;
        }
        if (extraction.update_fields?.includes('start_time')) {
          updates.start_time = extraction.start_time === 'null' ? null : extraction.start_time;
        }
        if (extraction.update_fields?.includes('end_time')) {
          updates.end_time = extraction.end_time === 'null' ? null : extraction.end_time;
        }

        if (shouldUpdateEmbedding) {
          const updatedText = [updates.content, updates.description].filter(Boolean).join(' ');
          updates.embedding = await generateEmbedding(updatedText);
        }

        const { data: updatedTask, error: updateError } = await supabase
          .from('tasks')
          .update(updates)
          .eq('id', extraction.matched_task_id)
          .select()
          .single();

        if (updateError) throw updateError;

        // Update calendar event
        try {
          if (updatedTask.calendar_event_id) {
            const calendar = await getGoogleCalendarClient(senderId);
            await updateCalendarEvent(calendar, updatedTask.calendar_event_id, updatedTask);
          }
        } catch (calendarError) {
          console.error('Calendar event update failed:', calendarError);
        }

        return { updated: updatedTask };

      case 'complete':
        if (!extraction.matched_task_id) {
          // Handle the case where no task ID is provided for completion
          throw new Error('No task ID for completion');
        }

        const { data: completedTask, error: completeError } = await supabase
          .from('tasks')
          .update({ status: 'completed', completed_at: new Date().toISOString() })
          .eq('id', extraction.matched_task_id)
          .select()
          .single();

        if (completeError) throw completeError;

        // Mark calendar event as completed (or delete it)
        try {
          if (completedTask.calendar_event_id) {
            const calendar = await getGoogleCalendarClient(senderId);
            // You can either delete the event or update its title to show completion
            await updateCalendarEvent(calendar, completedTask.calendar_event_id, {
              ...completedTask,
              content: `✓ ${completedTask.content} (Completed)`
            });
          }
        } catch (calendarError) {
          console.error('Calendar event completion update failed:', calendarError);
        }

        return { completed: completedTask };

      case 'cancel':
        if (!extraction.matched_task_id) throw new Error('No task ID for cancellation');

        const { data: cancelledTask, error: cancelError } = await supabase
          .from('tasks')
          .update({ status: 'cancelled' })
          .eq('id', extraction.matched_task_id)
          .select()
          .single();

        if (cancelError) throw cancelError;

        // Delete calendar event
        try {
          if (cancelledTask.calendar_event_id) {
            const calendar = await getGoogleCalendarClient(senderId);
            await deleteCalendarEvent(calendar, cancelledTask.calendar_event_id);
          }
        } catch (calendarError) {
          console.error('Calendar event deletion failed:', calendarError);
        }

        return { cancelled: cancelledTask };

      default:
        console.log("Task cannot be derived because the message seems casual or normal conversation");
    }
  } catch (error) {
    console.error('Error executing task action:', error);
    throw error;
  }
}


// Create MCP Server
const server = new Server(
  {
    name: 'task-management-server',
    version: '1.0.0',
  },
  {
    capabilities: {
      tools: {},
    },
  }
);

// Enhanced tools with calendar integration
const TOOLS: Tool[] = [
  {
    name: 'process_task_message',
    description: 'Process a message to extract and manage tasks with Google Calendar integration (create, update, complete, cancel)',
    inputSchema: {
      type: 'object',
      properties: {
        message: {
          type: 'string',
          description: 'The message content to process for task extraction'
        },
        sender_id: {
          type: 'string',
          description: 'ID of the message sender'
        },
        receiver_id: {
          type: 'string',
          description: 'ID of the message receiver'
        }
      },
      required: ['message', 'sender_id', 'receiver_id']
    }
  },
  {
    name: 'get_tasks',
    description: 'Retrieve tasks for specific users',
    inputSchema: {
      type: 'object',
      properties: {
        sender_id: {
          type: 'string',
          description: 'ID of the task sender'
        },
        receiver_id: {
          type: 'string',
          description: 'ID of the task receiver'
        },
        status: {
          type: 'string',
          enum: ['pending', 'completed', 'cancelled', 'all'],
          description: 'Filter tasks by status',
          default: 'all'
        },
        limit: {
          type: 'number',
          description: 'Maximum number of tasks to return',
          default: 50
        }
      },
      required: ['sender_id', 'receiver_id']
    }
  },
  {
    name: 'sync_calendar_auth',
    description: 'Store or update Google Calendar authentication for a user',
    inputSchema: {
      type: 'object',
      properties: {
        user_id: {
          type: 'string',
          description: 'User ID to associate with calendar auth'
        },
        access_token: {
          type: 'string',
          description: 'Google OAuth access token'
        },
        refresh_token: {
          type: 'string',
          description: 'Google OAuth refresh token'
        },
        expires_at: {
          type: 'string',
          description: 'Token expiration time (ISO string)'
        }
      },
      required: ['user_id', 'access_token', 'refresh_token', 'expires_at']
    }
  }
];

// Handle tool listing
server.setRequestHandler(ListToolsRequestSchema, async () => {
  return { tools: TOOLS };
});

// Handle tool calls
server.setRequestHandler(CallToolRequestSchema, async (request) => {
  const { name, arguments: args } = request.params;

  try {
    switch (name) {
      case 'process_task_message':
        const { message, sender_id, receiver_id } = args as { message: string; sender_id: string; receiver_id: string };
        const result = await processTaskMessage(message, sender_id, receiver_id);
        return {
          content: [
            {
              type: 'text',
              text: JSON.stringify(result, null, 2)
            }
          ]
        };

      case 'get_tasks':
        if (!args || typeof args !== 'object') {
          throw new Error('Invalid arguments');
        }
        const getSenderId = args.sender_id;
        const getReceiverId = args.receiver_id;
        const status = args.status || 'all';
        const limit = typeof args.limit === 'number' ? args.limit : 50;
        
        let query = supabase
          .from('tasks')
          .select('*')
          .or(`and(sender_id.eq.${getSenderId},receiver_id.eq.${getReceiverId}),and(sender_id.eq.${getReceiverId},receiver_id.eq.${getSenderId})`)
          .order('created_at', { ascending: false })
          .limit(limit);

        if (status !== 'all') {
          query = query.eq('status', status);
        }

        const { data: tasks, error } = await query;
        
        if (error) throw error;
        
        return {
          content: [
            {
              type: 'text',
              text: JSON.stringify({ 
                tasks: tasks || [], 
                count: tasks?.length || 0 
              }, null, 2)
            }
          ]
        };

        case 'sync_calendar_auth':
        if (!args || typeof args !== 'object') {
          throw new Error('Invalid arguments');
        }
        
        const { user_id, access_token, refresh_token, expiry_date, token_type, scope } = args as {
          user_id: string;
          access_token: string;
          refresh_token: string;
          expiry_date: number;
          token_type?: string;
          scope?: string;
        };

        const { data: authData, error: authError } = await supabase
          .from('user_google_tokens')
          .upsert([
            {
              user_id,
              access_token,
              refresh_token,
              expiry_date: expiry_date.toString(),
              token_type: token_type || 'Bearer',
              scope: scope || 'https://www.googleapis.com/auth/calendar',
              updated_at: new Date().toISOString()
            }
          ])
          .select()
          .single();

        if (authError) throw authError;

        return {
          content: [{
            type: 'text',
            text: JSON.stringify({
              success: true,
              message: 'Google authentication synced successfully',
              user_id: authData.user_id
            }, null, 2)
          }]
        };

      case 'check_calendar_eligibility':
        if (!args || typeof args !== 'object') {
          throw new Error('Invalid arguments');
        }
        
        const checkUserId = args.user_id;
        const eligibility = await checkCalendarEligibility(checkUserId as string);
        
        return {
          content: [{
            type: 'text',
            text: JSON.stringify(eligibility, null, 2)
          }]
        };

      default:
        throw new Error(`Unknown tool: ${name}`);
    }
  } catch (error) {
    const errorMessage = error instanceof Error ? error.message : 'Unknown error occurred';
    return {
      content: [{
        type: 'text',
        text: JSON.stringify({
          error: errorMessage,
          success: false
        }, null, 2)
      }],
      isError: true
    };
  }
});

// Start the server
async function main() {
  const transport = new StdioServerTransport();
  await server.connect(transport);
  console.error('Enhanced Task Management MCP Server with Google Calendar running');

  // Start background workers
  startBackgroundWorker();
  startTokenRefreshMonitor();
}

async function startTokenRefreshMonitor() {
  setInterval(async () => {
    try {
      const now = Date.now();
      const threshold = now + 5 * 60 * 1000;
      
      const { data: expiringTokens, error } = await supabase
        .from('user_google_tokens')
        .select('user_id, refresh_token, expiry_date')
        .lt('expiry_date', threshold.toString())
        .not('refresh_token', 'is', null);

      if (error) {
        console.error('Error fetching expiring tokens:', error);
        return;
      }

      if (expiringTokens?.length) {
        console.log(`Refreshing ${expiringTokens.length} expiring tokens`);
        
        const oauth2Client = new OAuth2Client(
          GOOGLE_CLIENT_ID,
          GOOGLE_CLIENT_SECRET,
          GOOGLE_REDIRECT_URI
        );

        for (const token of expiringTokens) {
          try {
            oauth2Client.setCredentials({
              refresh_token: token.refresh_token
            });
            
            const { credentials } = await oauth2Client.refreshAccessToken();
            const newExpiry = credentials.expiry_date || (now + 3600000);
            
            await supabase
              .from('user_google_tokens')
              .update({
                access_token: credentials.access_token,
                expiry_date: newExpiry.toString(),
                updated_at: new Date().toISOString()
              })
              .eq('user_id', token.user_id);
              
            console.log(`Refreshed token for user ${token.user_id}`);
          } catch (refreshError) {
            console.error(`Failed to refresh token for user ${token.user_id}:`, refreshError);
          }
        }
      }
    } catch (error) {
      console.error('Token refresh monitoring error:', error);
    }
  }, 5 * 60 * 1000);
}

// Background worker for automatic processing
async function startBackgroundWorker() {
  const subscription = supabase
    .channel('messages_channel')
    .on(
      'postgres_changes',
      {
        event: 'INSERT',
        schema: 'public',
        table: 'messages'
      },
      async (payload) => {
        console.error('Auto-processing new message with calendar integration');
        const message = payload.new as Message;
        
        // Skip system-inserted messages
        if (message.is_system) {
          console.error('Skipping system-inserted message.');
          return;
        }
        
        // Check if a task already exists for this message
        const { data: existingTasks, error } = await supabase
          .from('tasks')
          .select('id')
          .eq('message_id', message.id)
          .limit(1);

        if (error) {
          console.error('Error checking for existing task:', error);
          return;
        }
        
        if (existingTasks && existingTasks.length > 0) {
          console.error('Task already exists for this message, skipping.');
          return;
        }

        await processTaskMessage(message.content, message.sender_id, message.receiver_id);
      }
    )
    .subscribe();

  console.error('Background subscription active with calendar integration');
}

main().catch((error) => {
  console.error('Server error:', error);
  process.exit(1);
});
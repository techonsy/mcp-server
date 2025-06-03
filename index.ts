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
 import * as dotenv from 'dotenv';
console.log('Loading environment variables from .env file');
dotenv.config();

// Initialize Supabase client
const supabase = createClient(
    process.env.SUPABASE_URL || '',
    process.env.SUPABASE_SERVICE_ROLE_KEY || ''
);
const defaultDate = new Date().toISOString().split('T')[0]; 
// Initialize Cohere client
const cohere = new CohereClient({
  token: process.env.COHERE_API_KEY || '',
});

// Initialize OpenAI client
const openai = new OpenAI({
  baseURL: "https://openrouter.ai/api/v1",
  apiKey: process.env.OPENROUTER_API_KEY || '',
});

// Google Calendar configuration
const GOOGLE_CLIENT_ID = process.env.GOOGLE_CLIENT_ID || '';
const GOOGLE_CLIENT_SECRET = process.env.GOOGLE_CLIENT_SECRET || '';
const GOOGLE_REDIRECT_URI = process.env.GOOGLE_REDIRECT_URI || '';

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

async function createCalendarEvent(calendar: any, task: Task,receiverId:string): Promise<string | null> {
  try {
    // Skip calendar event creation if no meaningful date/time info
    if (!task.start_date && !task.due_date && !task.start_time) {
      console.log('No date/time info for calendar event, skipping');
      return null;
    }

    // Helper function to convert timestamp to Date with better validation
    const parseTimestamp = (date:any, time:any, fallback:any) => {
      let baseDate;

      if (date) {
        baseDate = new Date(date);
        if (isNaN(baseDate.getTime())) {
          console.warn(`Invalid date: ${date}, using fallback`);
          baseDate = fallback || new Date();
        }
      } else {
        baseDate = fallback || new Date();
      }

      if (time && time !== 'null') {
        const [hours, minutes] = time.split(':').map(Number);
        if (!isNaN(hours) && !isNaN(minutes)) {
          baseDate.setHours(hours, minutes, 0, 0);
        }
      }

      return baseDate;
    };

    // Parse start and end times with validation
    const startDateTime = parseTimestamp(
      task.start_date || task.due_date || '',
      task.start_time || '',
      new Date() // fallback to now
    );

    const endDateTime = parseTimestamp(
      task.start_date || task.due_date || '',
      task.end_time || '',
      new Date(startDateTime.getTime() + 60 * 60 * 1000) // Default 1 hour duration
    );

    // Ensure end time is after start time
    if (endDateTime <= startDateTime) {
      endDateTime.setTime(startDateTime.getTime() + 60 * 60 * 1000);
    }

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

    console.log('Creating calendar event for receiver:', receiverId);
    const response = await calendar.events.insert({
      calendarId: 'primary',
      requestBody: event,
    });

    console.log('Calendar event created successfully for receiver:', receiverId, response.data.id);
    return response.data.id;
  } catch (error) {
    console.error('Error creating calendar event:', error);
    if (error) {
      console.error('Calendar API error details:', error);
    }
    return null;
  }
}




async function updateCalendarEvent(calendar: any, eventId: string, task: Task): Promise<boolean> {
  try {
    // Helper function to convert timestamp to Date
    const parseTimestamp = (timestamp: string | null, fallback?: Date): Date => {
      if (timestamp) {
        return new Date(timestamp);
      }
      return fallback || new Date();
    };

    // Parse start and end times
    const startDateTime = parseTimestamp(
      task.start_time || null, 
      task.due_date ? new Date(task.due_date) : new Date()
    );
    
    const endDateTime = parseTimestamp(
      task.end_time || null,
      new Date(startDateTime.getTime() + 60 * 60 * 1000) // Default 1 hour duration
    );

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

async function checkMessageRelevancy(messageContent: string, userId:string, threshold = 0.7) {
  try {
    // Fetch recent messages for the user
    const recentMessages = await findRecentTasks(userId, userId);

    // Generate embedding for the current message
    const currentEmbedding = await generateEmbedding(messageContent);

    // Analyze the recent messages to find relevancy using embeddings
    for (const msg of recentMessages) {
      if (msg.embedding) {
        const similarity = cosineSimilarity(currentEmbedding, msg.embedding);
        if (similarity > threshold) {
          return msg; // Return the relevant message
        }
      }
    }

    return null; // No relevancy found
  } catch (error) {
    console.error('Error checking message relevancy:', error);
    return null;
  }
}


// Enhanced processTaskMessage with calendar integration
async function processTaskMessage(content: string, senderId: string, receiverId: string) {
  try {
    // First check if message is relevant to existing tasks
    const relevantMessage = await checkMessageRelevancy(content, senderId);
    
    if (relevantMessage) {
      // Process as update/complete/cancel for existing task
      const taskExtraction = await extractTaskFromMessage(content, [], []);
      
      if (!taskExtraction) {
        return { success: false, message: "No task action detected" };
      }

      // Validate we have a matched task ID for update/complete/cancel actions
      if (['update', 'complete', 'cancel'].includes(taskExtraction.action)) {
        if (!taskExtraction.matched_task_id && relevantMessage.id) {
          taskExtraction.matched_task_id = relevantMessage.id;
        }
        
        if (!taskExtraction.matched_task_id) {
          return { 
            success: false, 
            message: "No matching task found for update/complete/cancel action" 
          };
        }

        return await executeTaskAction(
          taskExtraction, 
          senderId, 
          receiverId, 
          relevantMessage.id || ''
        );
      }
    }

    // Process as new task creation
    const taskExtraction = await extractTaskFromMessage(content, [], []);
    
    if (!taskExtraction || taskExtraction.action !== 'create') {
      return { success: false, message: "No task creation detected" };
    }

    // Create message first to ensure referential integrity
    const messageId = uuidv4();
    const { error: messageError } = await supabase
      .from('messages')
      .insert({
        id: messageId,
        content,
        sender_id: senderId,
        receiver_id: receiverId,
        created_at: new Date().toISOString(),
        is_task_created: true
      });

    if (messageError) {
      console.error('Failed to create message:', messageError);
      throw messageError;
    }

    // Now create the task with the valid message_id
    const result = await executeTaskAction(
      taskExtraction, 
      senderId, 
      receiverId, 
      messageId
    );

    return result;

  } catch (error) {
    console.error('Error processing task message:', error);
    return { 
      success: false, 
      error: error instanceof Error ? error.message : 'Unknown error occurred',
      details: error instanceof Error ? error.stack : undefined
    };
  }
}

// Enhanced extractTaskFromMessage with calendar fields
async function extractTaskFromMessage(
  currentMessage: string,
  recentTasks: Task[] = [],
  relevantTasks: TaskSimilarity[] = [],
): Promise<TaskExtraction | null> {
  try {
    const tasksContext = recentTasks.length > 0
      ? recentTasks.map((task, i) =>
          `Task ${i + 1} (ID: ${task.id}): "${task.content}" - Priority: ${task.priority}, Status: ${task.status}, Due: ${task.due_date || 'No deadline'}, Start: ${task.start_date || 'No start date'}`)
        .join('\n')
      : "No recent tasks found.";

    const relevantTasksContext = relevantTasks.length > 0
      ? relevantTasks.map((item, i) =>
          `Relevant Task ${i + 1} (ID: ${item.task.id}, Similarity: ${(item.similarity * 100).toFixed(1)}%):\nContent: "${item.task.content}"\nPriority: ${item.task.priority}, Status: ${item.task.status}\nReasons: ${item.reasons.join(', ')}`)
        .join('\n\n')
      : "No relevant tasks found.";

    const currentDate = new Date().toISOString().split('T')[0];
    const currentTime = new Date().toTimeString().slice(0, 5); // HH:MM format

    const completion = await openai.chat.completions.create({
      model: "meta-llama/llama-3.1-8b-instruct:free",
      messages: [
        {
          role: "system",
          content: `You are a task management AI. Analyze messages and extract task information.

CRITICAL: Return ONLY valid JSON. No explanations, no extra text, no markdown.

CURRENT DATE: ${currentDate}
CURRENT TIME: ${currentTime}

TASK DETECTION RULES:
- Look for action words: do, make, create, schedule, plan, remind, complete, finish, cancel, update
- Look for time indicators: today, tomorrow, next week, at 3pm, by Friday, etc.
- Look for objects/goals: meeting, call, presentation, email, etc.

DATE/TIME PARSING RULES:
- "today" = ${currentDate}
- "tomorrow" = ${new Date(Date.now() + 24*60*60*1000).toISOString().split('T')[0]}
- "next week" = ${new Date(Date.now() + 7*24*60*60*1000).toISOString().split('T')[0]}
- "at 3pm", "3:00", "15:00" = extract time in HH:MM format
- "by Friday", "due Friday" = set as due_date
- If no specific time mentioned but date is mentioned, use "09:00" as default start time
- If start time but no end time, add 1 hour to start time for end time

PRIORITY RULES:
- "urgent", "asap", "immediately" = "urgent"
- "important", "priority" = "high"
- "when you can", "sometime" = "low"
- Default = "medium"

RESPONSE FORMAT (JSON only):
{
  "task": "extracted task description",
  "priority": "low|medium|high|urgent",
  "confidence": 0.0-1.0,
  "description": "detailed description of what needs to be done",
  "due_date": "YYYY-MM-DD or null",
  "start_date": "YYYY-MM-DD or null",
  "start_time": "HH:MM or null",
  "end_time": "HH:MM or null",
  "action": "create|update|complete|cancel",
  "matched_task_id": "task_id or null",
  "update_fields": []
}

EXAMPLES:
- "Schedule a meeting tomorrow at 2pm" → start_date: tomorrow, start_time: "14:00", end_time: "15:00"
- "Remind me to call John by Friday" → due_date: next Friday, task: "call John"
- "I need to finish the report today" → due_date: today, task: "finish the report"
- "Let's have lunch at noon" → start_date: today, start_time: "12:00", end_time: "13:00"`,
        },
        {
          role: "user",
          content: `MESSAGE: "${currentMessage}"

RECENT TASKS:
${tasksContext}

RELEVANT TASKS:
${relevantTasksContext}

Extract task information and return ONLY the JSON object:`,
        }
      ],
      temperature: 0.1,
      max_tokens: 500
    });

    const responseContent = completion.choices[0]?.message?.content;
    if (!responseContent) {
      console.log('No response from LLM');
      return null;
    }

    // Clean the response - remove any non-JSON content
    let cleanedResponse = responseContent.trim();

    // Extract JSON if wrapped in code blocks
    const jsonMatch = cleanedResponse.match(/```(?:\:json)?\s*(\{[\s\S]*\})\s*```/);
    if (jsonMatch) {
      cleanedResponse = jsonMatch[1];
    }

    // Find the JSON object if there's extra text
    const jsonStart = cleanedResponse.indexOf('{');
    const jsonEnd = cleanedResponse.lastIndexOf('}');
    if (jsonStart !== -1 && jsonEnd !== -1 && jsonEnd > jsonStart) {
      cleanedResponse = cleanedResponse.substring(jsonStart, jsonEnd + 1);
    }

    console.log('Cleaned LLM response:', cleanedResponse);

    let parsed: TaskExtraction;
    try {
      parsed = JSON.parse(cleanedResponse);
    } catch (parseError) {
      console.error('JSON parse error:', parseError);
      console.error('Failed to parse:', cleanedResponse);

      // Return a default "no task detected" response
      return {
        task: '',
        priority: "low",
        confidence: 0.0,
        description: "No task action detected",
        due_date: null,
        start_date: null,
        start_time: null,
        end_time: null,
        action: "create",
        matched_task_id: '',
        update_fields: []
      };
    }

    // Post-process and validate the parsed result
    if (!parsed.action || !['create', 'update', 'complete', 'cancel'].includes(parsed.action)) {
      console.error('Invalid action:', parsed.action);
      return null;
    }

    if (parsed.action === 'create' && (!parsed.task || parsed.task === 'null' || parsed.task === null)) {
      console.log('No task content for create action');
      return null;
    }

    // Clean up null strings
    if (parsed.due_date === 'null' || parsed.due_date === '') parsed.due_date = null;
    if (parsed.start_date === 'null' || parsed.start_date === '') parsed.start_date = null;
    if (parsed.start_time === 'null' || parsed.start_time === '') parsed.start_time = null;
    if (parsed.end_time === 'null' || parsed.end_time === '') parsed.end_time = null;

    // Auto-assign matched_task_id for update/complete/cancel if not provided
    if (['update', 'complete', 'cancel'].includes(parsed.action) && !parsed.matched_task_id && relevantTasks.length > 0) {
      parsed.matched_task_id = relevantTasks[0].task.id;
      console.log(`Auto-assigned matched_task_id: ${parsed.matched_task_id}`);
    }

    // Ensure confidence is a number
    if (typeof parsed.confidence !== 'number') {
      parsed.confidence = 0.5;
    }

    // If we have date/time info, ensure start_date is set for calendar creation
    if ((parsed.start_time || parsed.end_time) && !parsed.start_date) {
      parsed.start_date = parsed.due_date || currentDate;
    }

    console.log('Task processing result:', {
      success: true,
      action: parsed.action,
      task: parsed.task,
      confidence: parsed.confidence,
      dates: {
        due_date: parsed.due_date,
        start_date: parsed.start_date,
        start_time: parsed.start_time,
        end_time: parsed.end_time
      }
    });

    return parsed;
  } catch (error) {
    console.error('Error extracting task:', error);

    // Return a safe default instead of null
    return {
      task: '',
      priority: "low",
      confidence: 0.0,
      description: "Error processing message",
      due_date: null,
      start_date: null,
      start_time: null,
      end_time: null,
      action: "create",
      matched_task_id: '',
      update_fields: []
    };
  }
}




// Enhanced executeTaskAction with proper timestamp handling
async function executeTaskAction(extraction: TaskExtraction, senderId: string, receiverId: string, messageId: string): Promise<any> {
  try {

    if (messageId) {
      const { data: message, error: messageError } = await supabase
        .from('messages')
        .select('id')
        .eq('id', messageId)
        .single();

      if (messageError || !message) {
        console.warn(`Message ${messageId} not found, proceeding without message reference`);
        messageId = ''; // Clear the invalid message_id
      }
    }

    const defaultDate = new Date().toISOString().split('T')[0]; // Get today's date in YYYY-MM-DD format

    // Helper function to format time properly for PostgreSQL timestamp
    function formatTimestamp(date: string | null, time: string | null): string | null {
      if (!date && !time) return null;
      
      // Use provided date or default to today
      const dateStr = date || defaultDate;
      
      // Handle time formatting
      let timeStr = '00:00:00';
      if (time && time !== 'null') {
        // If time is in HH:MM format, add seconds
        if (time.match(/^\d{1,2}:\d{2}$/)) {
          timeStr = `${time}:00`;
        } else if (time.match(/^\d{1,2}:\d{2}:\d{2}$/)) {
          timeStr = time;
        }
      }
      
      // Return full ISO timestamp
      return new Date(`${dateStr}T${timeStr}`).toISOString();
    }

    switch (extraction.action) {
      case 'create':
        // Generate embedding for the new task
        const newTaskText = [extraction.task, extraction.description].filter(Boolean).join(' ');
        const newTaskEmbedding = await generateEmbedding(newTaskText);

        // Format timestamps properly
        const dueDate = extraction.due_date === 'null' ? null : extraction.due_date;
        const startTimestamp = formatTimestamp(extraction.start_date, extraction.start_time);
        const endTimestamp = formatTimestamp(extraction.start_date, extraction.end_time);

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
          start_date: extraction.start_date === 'null' ? null : extraction.start_date,
          start_time: startTimestamp, // Now properly formatted as full timestamp
          end_time: endTimestamp,     // Now properly formatted as full timestamp
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
          const calendar = await getGoogleCalendarClient(receiverId);
          calendarEventId = await createCalendarEvent(calendar, createdTask,receiverId);

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
          updates.start_time = formatTimestamp(extraction.start_date, extraction.start_time);
        }
        if (extraction.update_fields?.includes('end_time')) {
          updates.end_time = formatTimestamp(extraction.start_date, extraction.end_time);
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
        const message = payload.new as Message;

        if (message.is_system || message.is_task_created) {
          console.log('Skipping system/task-created message:', message.id);
          return;
        }

        console.log('New message received for processing:', message.id);

        // Check if a task already exists for this message
        const { data: existingTasks, error: existingTaskError } = await supabase
          .from('tasks')
          .select('id')
          .eq('message_id', message.id)
          .limit(1);

        if (existingTaskError) {
          console.error('Error checking for existing task:', existingTaskError);
          return;
        }

        if (existingTasks && existingTasks.length > 0) {
          console.log('Task already exists for this message, skipping:', message.id);
          return;
        }

        console.log('Processing message:', message.id);
        try {
          const result = await processTaskMessage(message.content, message.sender_id, message.receiver_id);
          console.log('Task processing result:', result);
        } catch (error) {
          console.error('Error processing task message:', error);
        }
      }
    )
    .subscribe();

  console.log('Background subscription active with calendar integration');
}


main().catch((error) => {
  console.error('Server error:', error);
  process.exit(1);
});
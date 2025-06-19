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
  apiKey: process.env.OPENAI_API_KEY|| '',
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

async function createCalendarEvent(calendar: any, task: Task, receiverId: string): Promise<string | null> {
  try {
    // Skip calendar event creation if no meaningful date/time info
    if (!task.start_date && !task.due_date && !task.start_time) {
      console.log('No date/time info for calendar event, skipping');
      return null;
    }

    // Helper function to convert timestamp to Date with better validation
    const parseTimestamp = (date: any, time: any, fallback: any) => {
      let baseDate;
      
      // Parse the date first
      if (date && date !== 'null' && date !== null && date !== '') {
        baseDate = new Date(date);
        if (isNaN(baseDate.getTime())) {
          console.warn(`Invalid date: ${date}, using fallback`);
          baseDate = new Date(fallback || new Date());
        }
      } else {
        baseDate = new Date(fallback || new Date());
      }

      // Create a new date object to avoid modifying the original
      const resultDate = new Date(baseDate);
      
      // Parse and set the time if provided
      if (time && time !== 'null' && time !== null && time !== '') {
        console.log(`Parsing time: ${time}`);
        
        // Handle different time formats
        let hours, minutes;
        
        if (time.includes(':')) {
          const timeParts = time.split(':');
          hours = parseInt(timeParts[0], 10);
          minutes = parseInt(timeParts[1], 10);
        } else if (time.length === 4) {
          // Handle HHMM format
          hours = parseInt(time.substring(0, 2), 10);
          minutes = parseInt(time.substring(2, 4), 10);
        } else if (time.length <= 2) {
          // Handle just hours
          hours = parseInt(time, 10);
          minutes = 0;
        }
        
        if (typeof hours !== 'undefined' && typeof minutes !== 'undefined' && !isNaN(hours) && !isNaN(minutes) && hours >= 0 && hours <= 23 && minutes >= 0 && minutes <= 59) {
          resultDate.setHours(hours, minutes, 0, 0);
          console.log(`Set time to ${hours}:${minutes} on date ${resultDate.toISOString()}`);
        } else {
          console.warn(`Invalid time format: ${time}, hours: ${hours}, minutes: ${minutes}`);
        }
      }
      
      return resultDate;
    };

    console.log('Task data before parsing:', {
      start_date: task.start_date,
      due_date: task.due_date,
      start_time: task.start_time,
      end_time: task.end_time
    });

    // Parse start and end times with validation
    const startDateTime = parseTimestamp(
      task.start_date || task.due_date,
      task.start_time,
      new Date() // fallback to now
    );

    console.log('Parsed start time:', startDateTime.toISOString());

    // For end time, use the same date but different time
    const endDateTime = parseTimestamp(
      task.start_date || task.due_date,
      task.end_time,
      new Date(startDateTime.getTime() + 60 * 60 * 1000) // Default 1 hour duration
    );

    console.log('Parsed end time:', endDateTime.toISOString());

    // Ensure end time is after start time
    if (endDateTime <= startDateTime) {
      endDateTime.setTime(startDateTime.getTime() + 60 * 60 * 1000);
      console.log('Adjusted end time to be 1 hour after start:', endDateTime.toISOString());
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

    console.log('Final event object:', JSON.stringify(event, null, 2));
    console.log('Creating calendar event for receiver:', receiverId);
    
    const response = await calendar.events.insert({
      calendarId: 'primary',
      requestBody: event,
    });

    console.log('Calendar event created successfully for receiver:', receiverId, response.data.id);
    return response.data.id;
    
  } catch (error) {
    console.error('Error creating calendar event:', error);
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
): Promise<TaskExtraction> {
  try {
    // Enhanced NLP preprocessing
    const preprocessedMessage = await enhanceMessageUnderstanding(currentMessage);

    // Build context with organizational awareness
    const context = buildOrganizationalContext(recentTasks, relevantTasks);

    // Get current date/time for reference
    const { currentDate, currentTime } = getCurrentDateTime();

    // Call OpenAI with structured output
    const extraction = await getStructuredTaskExtraction({
      message: preprocessedMessage,
      context,
      currentDate,
      currentTime,
      relevantTasks
    });

    // Validate and normalize the response
    return normalizeTaskExtraction(extraction, relevantTasks, currentDate);
  } catch (error) {
    console.error('Task extraction error:', error);
    return getDefaultTaskExtraction('Error processing message');
  }
}

// Helper functions

async function enhanceMessageUnderstanding(message: string): Promise<string> {
  // Step 1: Resolve organizational references
  const withReferences = await resolveOrganizationalReferences(message);
  
  // Step 2: Clarify ambiguous terms
  const clarified = await disambiguateTerms(withReferences);
  
  // Step 3: Normalize temporal expressions
  const withNormalizedTime = await normalizeTemporalExpressions(clarified);
  
  return withNormalizedTime;
}

async function resolveOrganizationalReferences(message: string): Promise<string> {
  try {
    const response = await openai.chat.completions.create({
      model: "gpt-4-turbo",
      messages: [{
        role: "system",
        content: `Resolve all organizational references in this message:
- Expand acronyms
- Replace pronouns with proper nouns
- Clarify department-specific terms
Return only the clarified message.`
      }, {
        role: "user",
        content: message
      }],
      temperature: 0
    });
    return response.choices[0]?.message?.content || message;
  } catch {
    return message;
  }
}

async function disambiguateTerms(message: string): Promise<string> {
  try {
    const response = await openai.chat.completions.create({
      model: "gpt-3.5-turbo",
      messages: [{
        role: "system",
        content: "Identify and disambiguate any potentially confusing terms in this organizational message. Return only the clarified message."
      }, {
        role: "user",
        content: message
      }],
      temperature: 0
    });
    return response.choices[0]?.message?.content || message;
  } catch {
    return message;
  }
}

async function normalizeTemporalExpressions(message: string): Promise<string> {
  try {
    const response = await openai.chat.completions.create({
      model: "gpt-3.5-turbo",
      messages: [{
        role: "system",
        content: `Normalize all time expressions in this message to unambiguous forms:
- "next week" → "the week of [date]"
- "EOQ" → "end of quarter (March 31/June 30/Sept 30/Dec 31)"
- "tomorrow afternoon" → "tomorrow at 2pm"
Return only the normalized message.`
      }, {
        role: "user",
        content: message
      }],
      temperature: 0
    });
    return response.choices[0]?.message?.content || message;
  } catch {
    return message;
  }
}

function buildOrganizationalContext(recentTasks: Task[], relevantTasks: TaskSimilarity[]): string {
  const recentTasksText = recentTasks.length > 0
    ? `RECENT TASKS:\n${recentTasks.map(task => 
        `- ${task.content} [ID:${task.id}, Status:${task.status}, Due:${task.due_date || 'none'}]`
      ).join('\n')}`
    : 'No recent tasks';

  const relevantTasksText = relevantTasks.length > 0
    ? `RELEVANT TASKS:\n${relevantTasks.map(item =>
        `- ${item.task.content} [ID:${item.task.id}, Match:${(item.similarity * 100).toFixed(0)}%, Reasons:${item.reasons.join('; ')}]`
      ).join('\n')}`
    : 'No relevant tasks';

  return `${recentTasksText}\n\n${relevantTasksText}`;
}

function getCurrentDateTime(): { currentDate: string; currentTime: string } {
  const now = new Date();
  return {
    currentDate: now.toISOString().split('T')[0],
    currentTime: now.toTimeString().slice(0, 5)
  };
}

async function getStructuredTaskExtraction(params: {
  message: string;
  context: string;
  currentDate: string;
  currentTime: string;
  relevantTasks: TaskSimilarity[];
}): Promise<Partial<TaskExtraction>> {
  const response = await openai.chat.completions.create({
    model: "gpt-4-turbo",
    response_format: { type: "json_object" },
    messages: [
      {
        role: "system",
        content: `You are an organizational task extraction system. Analyze messages and return JSON with:
- task: Clear action description
- priority: Based on organizational standards
- dates/times: Precisely normalized
- action: create/update/complete/cancel
- references: To existing tasks when applicable

Current Date: ${params.currentDate}
Current Time: ${params.currentTime}

Response must match this exact JSON structure:
{
  "task": string,
  "priority": "low"|"medium"|"high"|"urgent",
  "confidence": number(0-1),
  "description": string,
  "due_date": string|null,
  "start_date": string|null,
  "start_time": string|null,
  "end_time": string|null,
  "action": "create"|"update"|"complete"|"cancel",
  "existing_task_reference": string|null,
  "matched_task_id": string|null,
  "update_fields": string[]|null
}`
      },
      {
        role: "user",
        content: `MESSAGE: ${params.message}\n\nCONTEXT:\n${params.context}`
      }
    ],
    temperature: 0.2,
    max_tokens: 1000
  });

  try {
    return JSON.parse(response.choices[0]?.message?.content || '{}');
  } catch {
    return {};
  }
}

function normalizeTaskExtraction(
  extraction: Partial<TaskExtraction>,
  relevantTasks: TaskSimilarity[],
  currentDate: string
): TaskExtraction {
  // Apply defaults for required fields
  const normalized: TaskExtraction = {
    task: extraction.task || '',
    priority: extraction.priority || 'medium',
    confidence: Math.min(1, Math.max(0, extraction.confidence || 0.5)),
    description: extraction.description || '',
    due_date: extraction.due_date || null,
    start_date: extraction.start_date || null,
    start_time: extraction.start_time || null,
    end_time: extraction.end_time || null,
    action: extraction.action || 'create',
    existing_task_reference: extraction.existing_task_reference || undefined,
    matched_task_id: extraction.matched_task_id || undefined,
    update_fields: extraction.update_fields || undefined
  };

  // Auto-match relevant tasks for update/complete/cancel actions
  if (['update', 'complete', 'cancel'].includes(normalized.action) && 
      !normalized.matched_task_id && 
      relevantTasks.length > 0) {
    normalized.matched_task_id = relevantTasks[0].task.id;
  }

  // Ensure temporal consistency
  if ((normalized.start_time || normalized.end_time) && !normalized.start_date) {
    normalized.start_date = normalized.due_date || currentDate;
  }

  // Clean empty strings
  if (normalized.task.trim() === '') {
    normalized.confidence = 0;
  }

  return normalized;
}

function getDefaultTaskExtraction(reason: string): TaskExtraction {
  return {
    task: '',
    priority: 'medium',
    confidence: 0,
    description: reason,
    due_date: null,
    start_date: null,
    start_time: null,
    end_time: null,
    action: 'create'
  };
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
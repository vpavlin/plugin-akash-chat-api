import { createOpenAI } from '@ai-sdk/openai';
//import { getProviderBaseURL } from '@elizaos/core';
import type { ObjectGenerationParams, Plugin, TextEmbeddingParams } from '@elizaos/core';
import { type GenerateTextParams, ModelType, logger } from '@elizaos/core';
import { generateText } from 'ai';

/**
 * Retrieves a configuration setting from the runtime, environment variables, or a default value.
 *
 * Checks the runtime for the specified {@link key}, then environment variables, and finally returns {@link defaultValue} if neither is set.
 *
 * @param runtime - The runtime context providing configuration access.
 * @param key - The name of the setting to retrieve.
 * @param defaultValue - The value to return if the setting is not found.
 * @returns The setting value, or {@link defaultValue} if not found.
 */
function getSetting(runtime: any, key: string, defaultValue?: string): string | undefined {
  return runtime.getSetting(key) ?? process.env[key] ?? defaultValue;
}

/**
 * Returns the Akach Chat API base URL from runtime settings, environment variables, or a default value.
 *
 * @param runtime - The runtime context containing configuration settings.
 * @returns The resolved Akach Chat API base URL.
 */
function getBaseURL(runtime: any): string {
  const defaultBaseURL = getSetting(
    runtime,
    'AKASH_BASE_URL',
    process.env.AKASH_BASE_URL || 'https://chatapi.akash.network/api/v1'
  );
  return defaultBaseURL;//getProviderBaseURL(runtime, 'akashChatApi', defaultBaseURL);
}

/**
 * Retrieves the Akash Chat API key from runtime settings or environment variables.
 *
 * @returns The Akash Chat API key if available; otherwise, undefined.
 */
function getAkashChatApiKey(runtime: any): string | undefined {
  return getSetting(runtime, 'AKASH_API_KEY');
}

function getSmallModel(runtime: any): string {
  return getSetting(runtime, 'AKASH_SMALL_MODEL') ?? 'llama-3.3-70b';
}

function getLargeModel(runtime: any): string {
  return getSetting(runtime, 'AKASH_LARGE_MODEL') ?? 'llama-3.1-405b';
}

function getEmbeddingModel(runtime: any): string {
  return getSetting(runtime, 'AKASH_EMBEDDING_MODEL') ?? 'text-embedding-3-small';
}

function getEmbeddingDimensions(runtime: any): number {
  return parseInt(getSetting(runtime, 'AKASH_EMBEDDING_DIMENSIONS') ?? '4096', 10);
}

// OpenAI-specific helper functions
function getOpenAIApiKey(runtime: any): string | undefined {
  return getSetting(runtime, 'OPENAI_API_KEY');
}

function getOpenAIEmbeddingModel(runtime: any): string {
  return getSetting(runtime, 'OPENAI_EMBEDDING_MODEL') ?? 'text-embedding-3-small';
}

/**
 * Retrieves the configured embedding dimensions for OpenAI embeddings from runtime settings.
 *
 * @returns The number of embedding dimensions if set, or `undefined` if not configured.
 */
function getOpenAIEmbeddingDimensions(runtime: any): number | undefined {
  const dimsString = getSetting(runtime, 'OPENAI_EMBEDDING_DIMENSIONS');
  return dimsString ? parseInt(dimsString, 10) : undefined;
}

/**
 * Retrieves the OpenAI API base URL from runtime settings, environment variables, or defaults to 'https://api.openai.com/v1'.
 *
 * @returns The resolved OpenAI API base URL.
 */
function getOpenAIBaseURL(runtime: any): string {
  return 'https://api.openai.com/v1';
}

/**
 * Creates an OpenAI-compatible client configured for the Akach Chat API using the provided runtime context.
 *
 * @returns An OpenAI client instance set up with the Akash Chat API key and base URL.
 */
function createAkashChatApiClient(runtime: any) {
  return createOpenAI({
    apiKey: getAkashChatApiKey(runtime),
    baseURL: getBaseURL(runtime),
  });
}

const PLUGIN_VERSION = '1.1.2-obj-gen-fix'; // Updated version

export const akashPlugin: Plugin = {
  name: 'akashChatApi',
  description: `Akash Chat AI plugin (Handles Inference; Embeddings via OpenAI - v${PLUGIN_VERSION})`,
  config: {
    AKASH_API_KEY: process.env.AKASH_API_KEY,
    AKASH_SMALL_MODEL: process.env.AKASH_SMALL_MODEL,
    AKASH_LARGE_MODEL: process.env.AKASH_LARGE_MODEL,
    OPENAI_API_KEY: process.env.OPENAI_API_KEY,
    OPENAI_EMBEDDING_MODEL: process.env.OPENAI_EMBEDDING_MODEL,
    OPENAI_EMBEDDING_DIMENSIONS: process.env.OPENAI_EMBEDDING_DIMENSIONS,
  },
  async init(_config, runtime) {
    logger.info(`[plugin-akash-chat-api] Initializing v${PLUGIN_VERSION}`);
    if (!getAkashChatApiKey(runtime)) {
      logger.warn('[plugin-akash-chat-api] AKASH_API_KEY is not set - Akash Chat API text generation will fail');
    }
    if (!getOpenAIApiKey(runtime)) {
      logger.warn('[plugin-akash-chat-api] OPENAI_API_KEY is not set - Embeddings via OpenAI will fail');
    }
  },
  models: {
    [ModelType.TEXT_LARGE]: async (
      runtime,
      {
        prompt,
        stopSequences = [],
        maxTokens = 8192,
        temperature = 0.7,
        frequencyPenalty = 0.7,
        presencePenalty = 0.7,
      }: GenerateTextParams
    ) => {
      const akash = createAkashChatApiClient(runtime);
      const model = getLargeModel(runtime);
      logger.log(`[Akash Chat API] Using TEXT_LARGE model: ${model}`);

      const { text: akashResponse } = await generateText({
        model: akash.languageModel(model),
        prompt: prompt,
        system: runtime.character.system ?? undefined,
        temperature: temperature,
        maxTokens: maxTokens,
        frequencyPenalty: frequencyPenalty,
        presencePenalty: presencePenalty,
        stopSequences: stopSequences,
      });

      return akashResponse;
    },
    [ModelType.TEXT_SMALL]: async (
      runtime,
      {
        prompt,
        stopSequences = [],
        maxTokens = 8192,
        temperature = 0.7,
        frequencyPenalty = 0.7,
        presencePenalty = 0.7,
      }: GenerateTextParams
    ) => {
      const akash = createAkashChatApiClient(runtime);
      const model = getSmallModel(runtime);
      logger.log(`[Akash Chat API] Using TEXT_SMALL model: ${model}`);

      const { text: akashResponse } = await generateText({
        model: akash.languageModel(model),
        prompt: prompt,
        system: runtime.character.system ?? undefined,
        temperature: temperature,
        maxTokens: maxTokens,
        frequencyPenalty: frequencyPenalty,
        presencePenalty: presencePenalty,
        stopSequences: stopSequences,
      });

      return akashResponse;
    },
    [ModelType.OBJECT_LARGE]: async (runtime, params: ObjectGenerationParams) => {
      logger.debug(`[plugin-akash-chat-api v${PLUGIN_VERSION}] OBJECT_LARGE handler using generateText`);
      const akash = createAkashChatApiClient(runtime);
      const model = getLargeModel(runtime);
      logger.log(`[Akash Chat API] Using OBJECT_LARGE model: ${model}`);
      const jsonPrompt = `${params.prompt}\n\nPlease provide your response strictly in JSON format. Do not include any explanatory text before or after the JSON object.`;

      try {
        const { text: jsonText } = await generateText({
          model: akash.languageModel(model),
          prompt: jsonPrompt,
          temperature: params.temperature ?? 0, // Use lower temp for structured output
          // Note: No response_format parameter is sent here by generateText
        });

        // Log the raw text received BEFORE parsing
        logger.debug(
          `[plugin-akash-chat-api v${PLUGIN_VERSION}] Raw OBJECT_LARGE text received:`,
          jsonText
        );

        try {
          // Attempt to parse the result as JSON
          const object = JSON.parse(jsonText);
          logger.debug(
            `[plugin-akash-chat-api v${PLUGIN_VERSION}] Successfully parsed JSON from OBJECT_LARGE`
          );
          return object;
        } catch (parseError) {
          logger.error(
            `[plugin-akash-chat-api v${PLUGIN_VERSION}] Failed to parse JSON from OBJECT_LARGE response:`,
            { jsonText, parseError }
          );
          throw new Error('Model did not return valid JSON.');
        }
      } catch (error) {
        logger.error(
          `[plugin-akash-chat-api v${PLUGIN_VERSION}] Error during OBJECT_LARGE generation via generateText:`,
          error
        );
        throw error;
      }
    },
    [ModelType.OBJECT_SMALL]: async (runtime, params: ObjectGenerationParams) => {
      logger.debug(`[plugin-akash-chat-api v${PLUGIN_VERSION}] OBJECT_SMALL handler using generateText`);
      const akash = createAkashChatApiClient(runtime);
      const model = getSmallModel(runtime);
      logger.log(`[Akash Chat API] Using OBJECT_SMALL model: ${model}`);
      const jsonPrompt = `${params.prompt}\n\nPlease provide your response strictly in JSON format. Do not include any explanatory text before or after the JSON object.`;

      try {
        const { text: jsonText } = await generateText({
          model: akash.languageModel(model),
          prompt: jsonPrompt,
          temperature: params.temperature ?? 0, // Use lower temp for structured output
          // Note: No response_format parameter is sent here by generateText
        });

        // Log the raw text received BEFORE parsing
        logger.debug(
          `[plugin-akash-chat-api v${PLUGIN_VERSION}] Raw OBJECT_SMALL text received:`,
          jsonText
        );

        try {
          // Attempt to parse the result as JSON
          const object = JSON.parse(jsonText);
          logger.debug(
            `[plugin-akash-chat-api v${PLUGIN_VERSION}] Successfully parsed JSON from OBJECT_SMALL`
          );
          return object;
        } catch (parseError) {
          logger.error(
            `[plugin-akash-chat-api v${PLUGIN_VERSION}] Failed to parse JSON from OBJECT_SMALL response:`,
            { jsonText, parseError }
          );
          throw new Error('Model did not return valid JSON.');
        }
      } catch (error) {
        logger.error(
          `[plugin-akash-chat-api v${PLUGIN_VERSION}] Error during OBJECT_SMALL generation via generateText:`,
          error
        );
        throw error;
      }
    },
    [ModelType.TEXT_EMBEDDING]: async (runtime, params: TextEmbeddingParams): Promise<number[]> => {
      logger.debug(`[plugin-akash-chat-api/OpenAI Embed v${PLUGIN_VERSION}] Handler entered.`);
      const openaiApiKey = getOpenAIApiKey(runtime);
      const model = getOpenAIEmbeddingModel(runtime);
      logger.log(`[Akash Chat API/OpenAI Embed] Using TEXT_EMBEDDING model: ${model}`);
      const dimensions = getOpenAIEmbeddingDimensions(runtime);
      const hardcodedDimensionFallback = 1536;

      if (!params?.text || params.text.trim() === '') {
        logger.debug(
          `[plugin-akash-chat-api/OpenAI Embed v${PLUGIN_VERSION}] Creating test embedding for initialization/empty text`
        );
        const initDimensions = dimensions ?? hardcodedDimensionFallback;
        const testVector = new Array(initDimensions).fill(0);
        testVector[0] = 0.1;
        return testVector;
      }

      if (!openaiApiKey) {
        logger.error(
          `[plugin-akash-chat-api/OpenAI Embed v${PLUGIN_VERSION}] OPENAI_API_KEY is missing. Cannot generate embedding.`
        );
        const errorDims = dimensions ?? hardcodedDimensionFallback;
        const errorVector = new Array(errorDims).fill(0);
        errorVector[0] = 0.3;
        return errorVector;
      }

      logger.debug(`[plugin-akash-chat-api/OpenAI Embed v${PLUGIN_VERSION}] Attempting OpenAI API call...`);
      try {
        const payload: { model: string; input: string; dimensions?: number } = {
          model: model,
          input: params.text,
        };
        if (dimensions !== undefined) {
          payload.dimensions = dimensions;
        }
        logger.debug(
          `[plugin-akash-chat-api/OpenAI Embed v${PLUGIN_VERSION}] Calling ${getOpenAIBaseURL(runtime)}/embeddings with model ${model}`
        );

        const response = await fetch(`${getOpenAIBaseURL(runtime)}/embeddings`, {
          method: 'POST',
          headers: {
            Authorization: `Bearer ${openaiApiKey}`,
            'Content-Type': 'application/json',
          },
          body: JSON.stringify(payload),
        });

        logger.debug(
          `[plugin-akash-chat-api/OpenAI Embed v${PLUGIN_VERSION}] Received response status: ${response.status}`
        );
        if (!response.ok) {
          let errorBody = 'Could not parse error body';
          try {
            errorBody = await response.text();
          } catch (e) {
            /* ignore */
          }
          logger.error(
            `[plugin-akash-chat-api/OpenAI Embed v${PLUGIN_VERSION}] OpenAI API error: ${response.status} - ${response.statusText}`,
            { errorBody }
          );
          throw new Error(`OpenAI API error: ${response.status} - ${response.statusText}`);
        }

        const data = (await response.json()) as {
          data: Array<{ embedding: number[] }>;
          usage: object;
          model: string;
        };
        logger.debug(
          `[plugin-akash-chat-api/OpenAI Embed v${PLUGIN_VERSION}] Successfully parsed OpenAI response.`
        );

        if (!data?.data?.[0]?.embedding) {
          logger.error(
            `[plugin-akash-chat-api/OpenAI Embed v${PLUGIN_VERSION}] No embedding returned from OpenAI API`,
            { responseData: data }
          );
          throw new Error('No embedding returned from OpenAI API');
        }

        const embedding = data.data[0].embedding;
        const embeddingDimensions = embedding.length;

        if (dimensions !== undefined && embeddingDimensions !== dimensions) {
          logger.warn(
            `[plugin-akash-chat-api/OpenAI Embed v${PLUGIN_VERSION}] OpenAI Embedding dimensions mismatch: requested ${dimensions}, got ${embeddingDimensions}`
          );
        }

        logger.debug(
          `[plugin-akash-chat-api/OpenAI Embed v${PLUGIN_VERSION}] Returning embedding with dimensions ${embeddingDimensions}.`
        );
        return embedding;
      } catch (error) {
        logger.error(
          `[plugin-akash-chat-api/OpenAI Embed v${PLUGIN_VERSION}] Error during OpenAI embedding generation process:`,
          error
        );
        const errorDims = dimensions ?? hardcodedDimensionFallback;
        const errorVector = new Array(errorDims).fill(0);
        errorVector[0] = 0.2;
        return errorVector;
      }
    },
  },
};

export default akashPlugin;

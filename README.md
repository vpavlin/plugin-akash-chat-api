# @vpavlin/plugin-akash-chat-api


## Configuration

The plugin requires the following environment variables to be set:

- `AKASH_CHAT_API_KEY`: Your Venice AI API key
- `AKASH_SMALL_MODEL`: The model to use for small text generation (defaults to 'default')
- `AKASH_LARGE_MODEL`: The model to use for large text generation (defaults to 'default')
- `AKASH_EMBEDDING_MODEL`: The model to use for embeddings (optional)
- `AKASH_EMBEDDING_DIMENSIONS`: The dimensions for embeddings (optional)


## Usage

To use the Venice plugin, add it to your ElizaOS configuration:

```typescript
import { akashPlugin } from '@vpavlin/plugin-achash-chat-api';

// Add to your plugins array
const plugins = [
  akashPlugin,
  // ... other plugins
];
```

## Development

To build the plugin:

```bash
bun run build
```

To watch for changes during development:

```bash
bun run dev
```

## License

MIT

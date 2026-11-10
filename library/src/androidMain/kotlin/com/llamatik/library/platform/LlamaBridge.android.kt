package com.llamatik.library.platform

import androidx.compose.runtime.Composable
import androidx.compose.ui.platform.LocalContext
import java.io.File

@Suppress("EXPECT_ACTUAL_CLASSIFIERS_ARE_IN_BETA_WARNING")
actual object LlamaBridge {
    init {
        System.loadLibrary("llama_jni") // Only here, where System exists
    }

    actual external fun initModel(modelPath: String): Boolean
    actual external fun embed(input: String): FloatArray

    @Composable
    actual fun getModelPath(modelFileName: String): String {
        val context = LocalContext.current
        val outFile = File(context.cacheDir, modelFileName)

        if (!outFile.exists()) {
            context.assets.open(modelFileName).use { inputStream ->
                outFile.outputStream().use { outputStream ->
                    inputStream.copyTo(outputStream)
                }
            }
        }

        return outFile.absolutePath
    }

    actual external fun initGenerateModel(modelPath: String): Boolean
    actual external fun generate(prompt: String): String
    actual external fun generateWithContext(systemPrompt: String, contextBlock: String, userPrompt: String): String
    private external fun nativeGenerateStream(prompt: String, callback: GenStream)
    private external fun nativeGenerateWithContextStream(system: String, context: String, user: String, callback: GenStream)

    actual fun generateStream(prompt: String, callback: GenStream) {
        nativeGenerateStream(prompt, callback)
    }

    // ---- Explicit context window and output limit (Android only, not in expect) ----

    /**
     * Same as [initGenerateModel], but with the context window chosen by the caller:
     * n_ctx = [nCtx] clamped to 512..32768, n_batch = min(2048, n_ctx); everything else
     * (n_ubatch, KV cache types, threads) stays at the llama.cpp defaults.
     */
    external fun initGenerateModelWithContext(modelPath: String, nCtx: Int): Boolean

    /**
     * Like [generate], but with an explicit output limit ([maxNewTokens], clamped to
     * 1..n_ctx) and without prompt truncation. Returns null when prompt tokens +
     * maxNewTokens > n_ctx (logcat: "prompt_too_long: <prompt> + <max> > <n_ctx>"),
     * and on the same failures where [generate] gets null from native code.
     */
    external fun generateWithLimit(prompt: String, maxNewTokens: Int): String?

    private external fun nativeGenerateStreamWithLimit(prompt: String, maxNewTokens: Int, callback: GenStream)

    /**
     * Like [generateStream], but with an explicit output limit ([maxNewTokens], clamped
     * to 1..n_ctx) and without prompt truncation. When prompt tokens + maxNewTokens >
     * n_ctx, only `callback.onError("prompt_too_long: <prompt> + <max> > <n_ctx>")` is
     * called (no onComplete), as for the other errors before the first token.
     */
    fun generateStreamWithLimit(prompt: String, maxNewTokens: Int, callback: GenStream) {
        nativeGenerateStreamWithLimit(prompt, maxNewTokens, callback)
    }

    // Compose the same chat prompt you use on native. This mirrors your Gemma-style tags
    // and matches the EOT checks we added in C++ (<start_of_turn>, <end_of_turn>, <|eot_id|>).
    private fun buildChatPrompt(systemPrompt: String, contextBlock: String, userPrompt: String): String {
        return buildString {
            append("<start_of_turn>system\n")
            append(systemPrompt.trim())
            append("\n<end_of_turn>\n")
            append("<start_of_turn>user\n")
            append("CONTEXT:\n")
            append(contextBlock.trim())
            append("\n\nQUESTION:\n")
            append(userPrompt.trim())
            append("\n<end_of_turn>\n")
            append("<start_of_turn>assistant\n")
        }
    }

    actual fun generateStreamWithContext(
        systemPrompt: String,
        contextBlock: String,
        userPrompt: String,
        callback: GenStream
    ) {
        val prompt = buildChatPrompt(systemPrompt, contextBlock, userPrompt)
        generateStream(prompt, callback)
    }

    actual fun generateWithContextStream(
        system: String,
        context: String,
        user: String,
        onDelta: (String) -> Unit,
        onDone: () -> Unit,
        onError: (String) -> Unit
    ) {
        val cb = object : GenStream {
            override fun onDelta(text: String) = onDelta(text)
            override fun onComplete() = onDone()
            override fun onError(message: String) = onError(message)
        }
        nativeGenerateWithContextStream(system, context, user, cb)
    }

    actual external fun shutdown()
}
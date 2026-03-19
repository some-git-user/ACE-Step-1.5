(() => {
  const STORAGE_KEY = "acestep.ui.audio.volume";
  const GENERATE_BUTTON_ID = "acestep-generate-btn";
  const STATUS_OUTPUT_ID = "acestep-status-output";
  const DEFAULT_VOLUME = 0.5;
  const EPSILON = 0.001;
  const STARTUP_RESYNC_WINDOW_MS = 3000;
  const STARTUP_RESYNC_INTERVAL_MS = 120;
  const STATUS_POLL_INTERVAL_MS = 400;
  const COMPLETION_SOUND_COOLDOWN_MS = 1000;
  const seenPlayers = new WeakSet();
  const seenVolumeSliders = new WeakSet();
  const sliderSyncSuppressedUntil = new WeakMap();
  const observedRoots = new WeakSet();
  const knownRoots = [];
  const readyForPersistence = new WeakMap();
  const wasReadyBeforeLoad = new WeakMap();
  let scanScheduled = false;
  let preferredVolume = null;
  let startupResyncTimer = null;
  let statusPollTimer = null;
  let latestStatusSnapshot = "";
  let lastCompletionSoundAt = 0;
  let completionAudioContext = null;

  const clampVolume = (value) => {
    if (value === null || value === undefined || value === "") {
      return null;
    }
    const parsed = Number(value);
    if (!Number.isFinite(parsed)) {
      return null;
    }
    if (parsed < 0) {
      return 0;
    }
    if (parsed > 1) {
      return 1;
    }
    return parsed;
  };

  const loadPreferredVolume = () => {
    try {
      const stored = window.localStorage.getItem(STORAGE_KEY);
      return clampVolume(stored);
    } catch (_error) {
      return null;
    }
  };

  const storePreferredVolume = (value) => {
    const clamped = clampVolume(value);
    if (clamped === null) {
      return;
    }
    preferredVolume = clamped;
    try {
      window.localStorage.setItem(STORAGE_KEY, String(clamped));
    } catch (_error) {
      // Ignore storage failures (private mode / blocked storage).
    }
  };

  const isTrustedUserEvent = (event) => Boolean(event && event.isTrusted);

  const applyPreferredVolume = (player) => {
    if (!player || preferredVolume === null) {
      return;
    }
    if (Math.abs(player.volume - preferredVolume) <= EPSILON) {
      return;
    }
    player.volume = preferredVolume;
  };

  const forEachAudioPlayer = (callback) => {
    for (let i = 0; i < knownRoots.length; i += 1) {
      const root = knownRoots[i];
      if (!root || !root.querySelectorAll) {
        continue;
      }
      root.querySelectorAll("audio").forEach((player) => callback(player));
    }
  };

  const forEachVolumeSlider = (callback) => {
    for (let i = 0; i < knownRoots.length; i += 1) {
      const root = knownRoots[i];
      if (!root || !root.querySelectorAll) {
        continue;
      }
      root
        .querySelectorAll(
          "input.volume-slider[type='range'], input#volume[type='range']",
        )
        .forEach((slider) => callback(slider));
    }
  };

  const applyPreferredVolumeToSlider = (slider) => {
    if (!slider || preferredVolume === null) {
      return;
    }
    const current = clampVolume(slider.value);
    if (current !== null && Math.abs(current - preferredVolume) <= EPSILON) {
      return;
    }
    sliderSyncSuppressedUntil.set(slider, Date.now() + 250);
    slider.value = String(preferredVolume);
    slider.dispatchEvent(new Event("input", { bubbles: true }));
  };

  const syncAllVolumeControlsToPreferred = (
    sourcePlayer = null,
    sourceSlider = null,
  ) => {
    if (preferredVolume === null) {
      return;
    }
    forEachAudioPlayer((player) => {
      if (!player) {
        return;
      }
      if (sourcePlayer && player !== sourcePlayer) {
        if (Math.abs(player.volume - preferredVolume) > EPSILON) {
          player.volume = preferredVolume;
        }
        return;
      }
      applyPreferredVolume(player);
    });
    forEachVolumeSlider((slider) => {
      if (!slider || slider === sourceSlider) {
        return;
      }
      applyPreferredVolumeToSlider(slider);
    });
  };

  const stopAndRewindAllPlayers = () => {
    discoverRoots();
    forEachAudioPlayer((player) => {
      if (!player) {
        return;
      }
      try {
        player.pause();
      } catch (_error) {
        // Ignore pause failures from detached or blocked media elements.
      }
      if (player.currentTime > 0) {
        player.currentTime = 0;
      }
      applyPreferredVolume(player);
    });
  };

  const isGenerateButtonClick = (event) => {
    if (!event) {
      return false;
    }
    const selector = `#${GENERATE_BUTTON_ID}`;
    const target = event.target;
    if (
      target &&
      typeof target.closest === "function" &&
      target.closest(selector)
    ) {
      return true;
    }
    if (typeof event.composedPath !== "function") {
      return false;
    }
    const path = event.composedPath();
    for (let i = 0; i < path.length; i += 1) {
      const node = path[i];
      if (node && node.id === GENERATE_BUTTON_ID) {
        return true;
      }
    }
    return false;
  };

  const handleDocumentClick = (event) => {
    if (!isGenerateButtonClick(event)) {
      return;
    }
    stopAndRewindAllPlayers();
  };

  const stopStartupResync = () => {
    if (startupResyncTimer === null) {
      return;
    }
    window.clearInterval(startupResyncTimer);
    startupResyncTimer = null;
  };

  const beginStartupResync = () => {
    stopStartupResync();
    if (preferredVolume === null) {
      return;
    }

    const startedAt = Date.now();
    startupResyncTimer = window.setInterval(() => {
      scanPlayers();
      syncAllVolumeControlsToPreferred();
      if (Date.now() - startedAt >= STARTUP_RESYNC_WINDOW_MS) {
        stopStartupResync();
      }
    }, STARTUP_RESYNC_INTERVAL_MS);
  };

  const observeRoot = (root) => {
    if (!root || observedRoots.has(root)) {
      return;
    }
    observedRoots.add(root);
    knownRoots.push(root);

    const observeTarget = root === document ? document.documentElement : root;
    if (!observeTarget) {
      return;
    }

    new MutationObserver(() => {
      scheduleScan();
    }).observe(observeTarget, { childList: true, subtree: true });
  };

  const forEachStatusOutputInput = (callback) => {
    const selector = `#${STATUS_OUTPUT_ID} textarea, #${STATUS_OUTPUT_ID} input[type='text'], #${STATUS_OUTPUT_ID} input:not([type])`;
    for (let i = 0; i < knownRoots.length; i += 1) {
      const root = knownRoots[i];
      if (!root || !root.querySelectorAll) {
        continue;
      }
      root.querySelectorAll(selector).forEach((node) => callback(node));
    }
  };

  const getStatusOutputText = () => {
    let value = "";
    forEachStatusOutputInput((node) => {
      if (value) {
        return;
      }
      if (typeof node.value === "string" && node.value.trim()) {
        value = node.value.trim();
      } else if (
        typeof node.textContent === "string" &&
        node.textContent.trim()
      ) {
        value = node.textContent.trim();
      }
    });
    return value;
  };

  const hasFailureMarker = (statusTextLower) =>
    statusTextLower.includes("❌") ||
    statusTextLower.includes("error") ||
    statusTextLower.includes("failed") ||
    statusTextLower.includes("traceback");

  const hasCompletionMarker = (statusTextLower) =>
    statusTextLower.includes("generation complete") ||
    statusTextLower.includes("created successfully") ||
    statusTextLower.includes("format_success") ||
    statusTextLower.includes("formatted") ||
    statusTextLower.includes("completed") ||
    statusTextLower.includes("success") ||
    statusTextLower.includes("done") ||
    statusTextLower.includes("ready");

  const isIntermediateStatus = (statusTextLower) =>
    statusTextLower.startsWith("encoding & ready:") ||
    statusTextLower.startsWith("preparing generation") ||
    statusTextLower.includes("autogen will generate next batch");

  const shouldPlayCompletionSound = (statusText) => {
    if (!statusText || typeof statusText !== "string") {
      return false;
    }
    const statusTextLower = statusText.toLowerCase();
    if (hasFailureMarker(statusTextLower)) {
      return false;
    }
    if (isIntermediateStatus(statusTextLower)) {
      return false;
    }
    return hasCompletionMarker(statusTextLower);
  };

  const getCompletionAudioContext = () => {
    if (completionAudioContext) {
      return completionAudioContext;
    }
    const AudioContextCtor = window.AudioContext || window.webkitAudioContext;
    if (!AudioContextCtor) {
      return null;
    }
    completionAudioContext = new AudioContextCtor();
    return completionAudioContext;
  };

  const playCompletionTone = () => {
    const audioContext = getCompletionAudioContext();
    if (!audioContext) {
      return;
    }

    const scheduleTone = (frequency, startAt, durationSeconds, gainValue) => {
      const oscillator = audioContext.createOscillator();
      const gainNode = audioContext.createGain();
      oscillator.type = "sine";
      oscillator.frequency.setValueAtTime(frequency, startAt);
      gainNode.gain.setValueAtTime(0.0001, startAt);
      gainNode.gain.exponentialRampToValueAtTime(gainValue, startAt + 0.01);
      gainNode.gain.exponentialRampToValueAtTime(
        0.0001,
        startAt + durationSeconds,
      );
      oscillator.connect(gainNode);
      gainNode.connect(audioContext.destination);
      oscillator.start(startAt);
      oscillator.stop(startAt + durationSeconds + 0.01);
    };

    const now = audioContext.currentTime + 0.01;
    scheduleTone(880, now, 0.08, 0.05);
    scheduleTone(1175, now + 0.12, 0.12, 0.06);
  };

  const handleStatusOutputChanged = () => {
    const statusText = getStatusOutputText();
    if (!statusText || statusText === latestStatusSnapshot) {
      return;
    }
    latestStatusSnapshot = statusText;

    if (!shouldPlayCompletionSound(statusText)) {
      return;
    }

    const nowMs = Date.now();
    if (nowMs - lastCompletionSoundAt < COMPLETION_SOUND_COOLDOWN_MS) {
      return;
    }
    lastCompletionSoundAt = nowMs;

    try {
      const ctx = getCompletionAudioContext();
      if (ctx && ctx.state === "suspended") {
        ctx
          .resume()
          .then(playCompletionTone)
          .catch(() => {
            // Ignore autoplay-policy resume errors.
          });
      } else {
        playCompletionTone();
      }
    } catch (_error) {
      // Completion sound is best-effort only.
    }
  };

  const startStatusPolling = () => {
    if (statusPollTimer !== null) {
      window.clearInterval(statusPollTimer);
      statusPollTimer = null;
    }
    statusPollTimer = window.setInterval(() => {
      discoverRoots();
      handleStatusOutputChanged();
    }, STATUS_POLL_INTERVAL_MS);
  };

  const stopStatusPolling = () => {
    if (statusPollTimer === null) {
      return;
    }
    window.clearInterval(statusPollTimer);
    statusPollTimer = null;
  };

  const scheduleScan = () => {
    if (scanScheduled) {
      return;
    }
    scanScheduled = true;
    requestAnimationFrame(() => {
      scanScheduled = false;
      scanPlayers();
    });
  };

  const discoverRoots = () => {
    const queue = [document];
    const visited = new WeakSet();

    while (queue.length > 0) {
      const root = queue.pop();
      if (!root || visited.has(root)) {
        continue;
      }
      visited.add(root);
      observeRoot(root);

      if (!root.querySelectorAll) {
        continue;
      }
      root.querySelectorAll("*").forEach((node) => {
        if (node && node.shadowRoot) {
          queue.push(node.shadowRoot);
        }
      });
    }
  };

  const registerPlayer = (player) => {
    if (!player || seenPlayers.has(player)) {
      return;
    }
    seenPlayers.add(player);
    readyForPersistence.set(player, player.readyState > 0);
    wasReadyBeforeLoad.set(player, false);
    applyPreferredVolume(player);

    player.addEventListener(
      "volumechange",
      (event) => {
        const next = clampVolume(player.volume);
        if (next === null) {
          return;
        }

        if (!isTrustedUserEvent(event)) {
          applyPreferredVolume(player);
          return;
        }

        if (
          preferredVolume !== null &&
          Math.abs(next - preferredVolume) <= EPSILON
        ) {
          return;
        }

        if (
          readyForPersistence.get(player) !== true &&
          wasReadyBeforeLoad.get(player) !== true
        ) {
          // Ignore mount/load reset events before media metadata is ready.
          applyPreferredVolume(player);
          return;
        }

        storePreferredVolume(next);
        syncAllVolumeControlsToPreferred(player, null);
      },
      { passive: true },
    );

    const markReadyForPersistence = () => {
      readyForPersistence.set(player, true);
      wasReadyBeforeLoad.set(player, false);
      applyPreferredVolume(player);
      if (player.currentTime > 0) {
        player.currentTime = 0;
      }
    };

    player.addEventListener(
      "loadedmetadata",
      () => {
        markReadyForPersistence();
      },
      { passive: true },
    );

    player.addEventListener(
      "loadstart",
      () => {
        wasReadyBeforeLoad.set(
          player,
          readyForPersistence.get(player) === true,
        );
        readyForPersistence.set(player, false);
        applyPreferredVolume(player);
        if (player.currentTime > 0) {
          player.currentTime = 0;
        }
      },
      { passive: true },
    );

    if (player.readyState > 0) {
      markReadyForPersistence();
    }
  };

  const registerVolumeSlider = (slider) => {
    if (!slider || seenVolumeSliders.has(slider)) {
      return;
    }
    seenVolumeSliders.add(slider);
    applyPreferredVolumeToSlider(slider);

    const onSliderInput = (event) => {
      const suppressedUntil = sliderSyncSuppressedUntil.get(slider) || 0;
      if (Date.now() <= suppressedUntil) {
        return;
      }

      if (!isTrustedUserEvent(event)) {
        applyPreferredVolumeToSlider(slider);
        return;
      }

      const next = clampVolume(slider.value);
      if (next === null) {
        return;
      }
      if (
        preferredVolume !== null &&
        Math.abs(next - preferredVolume) <= EPSILON
      ) {
        return;
      }
      storePreferredVolume(next);
      syncAllVolumeControlsToPreferred(null, slider);
    };

    slider.addEventListener("input", onSliderInput, { passive: true });
    slider.addEventListener("change", onSliderInput, { passive: true });
  };

  const scanPlayers = () => {
    discoverRoots();
    forEachAudioPlayer(registerPlayer);
    forEachVolumeSlider(registerVolumeSlider);
    syncAllVolumeControlsToPreferred();
  };

  const start = () => {
    preferredVolume = loadPreferredVolume();
    if (preferredVolume === null) {
      // First run (or invalid storage): seed a sane audible default.
      storePreferredVolume(DEFAULT_VOLUME);
      if (preferredVolume === null) {
        preferredVolume = DEFAULT_VOLUME;
      }
    }
    document.addEventListener("click", handleDocumentClick, true);
    scanPlayers();
    beginStartupResync();
    startStatusPolling();
    handleStatusOutputChanged();
    window.addEventListener(
      "beforeunload",
      () => {
        stopStartupResync();
        stopStatusPolling();
        document.removeEventListener("click", handleDocumentClick, true);
      },
      { once: true },
    );
  };

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", start, { once: true });
  } else {
    start();
  }
})();

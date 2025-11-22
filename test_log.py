# test_log.py — robust logger + diagnostics
import csv, os, time, traceback

LOG_CSV = "predictions_log.csv"

def log_prediction(input_name, pred_label, confidence, age_group):
    header = ["timestamp","input_file","pred_label","confidence","age_group"]
    now = time.strftime("%Y-%m-%d %H:%M:%S")
    row = [now, input_name, pred_label, confidence, age_group]

    try:
        # Ensure directory exists
        path = os.path.abspath(LOG_CSV)
        d = os.path.dirname(path)
        if d and not os.path.exists(d):
            os.makedirs(d, exist_ok=True)

        # If file exists and is empty, treat as not existing for header write
        needs_header = True
        if os.path.exists(path) and os.path.getsize(path) > 0:
            needs_header = False

        with open(path, "a", newline="", encoding="utf8") as f:
            writer = csv.writer(f)
            if needs_header:
                writer.writerow(header)
            writer.writerow(row)
            f.flush()
            try:
                os.fsync(f.fileno())
            except Exception:
                pass

        print("SUCCESS: Wrote log to:", path)
        return True
    except Exception as e:
        print("ERROR: Failed to write log.")
        print("Exception:", repr(e))
        print(traceback.format_exc())
        return False

def main():
    print("Current working directory:", os.getcwd())
    print("Absolute path of LOG_CSV will be:", os.path.abspath(LOG_CSV))
    print("Files in this folder:")
    for fn in os.listdir("."):
        print(" -", fn)

    print("\nAttempting to write test log now...")
    ok = log_prediction("test_run.wav", "Indian", "99.9%", "Adult")
    if not ok:
        print("Writing failed. Please copy the printed error above and paste here.")

    print("\nNow checking file exists and size...")
    p = os.path.abspath(LOG_CSV)
    if os.path.exists(p):
        print("File exists at:", p)
        try:
            print("File size (bytes):", os.path.getsize(p))
            print("\n--- File preview (first 10 lines) ---")
            with open(p, "r", encoding="utf8") as f:
                for i, line in enumerate(f):
                    if i >= 10:
                        break
                    print(line.rstrip())
        except Exception as e:
            print("Could not read file:", e)
    else:
        print("File not found at expected path:", p)
        print("Possible causes: OneDrive/permission blocking, antivirus, or running in a different folder than you think.")

if __name__ == "__main__":
    main()


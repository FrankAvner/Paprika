from pathlib import Path
import sqlite3


# ============================================================
# PAPRIKA - Leaf Tracking Database Migration
# ============================================================
#
# Updates the existing SQLite database without deleting data.
#
# Changes:
#
# segmentation_runs:
#   detected_leaf_count
#   selected_leaf_count
#   selection_created_at
#
# leaf_objects:
#   check_status
#   checked_at
#   check_count
#
# Existing data is preserved.
# ============================================================


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATABASE_PATH = PROJECT_ROOT / "data" / "paprika.db"


def get_connection():
    connection = sqlite3.connect(str(DATABASE_PATH))
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    return connection


def get_columns(connection, table_name):
    rows = connection.execute(
        f"PRAGMA table_info({table_name})"
    ).fetchall()

    return {
        row["name"]
        for row in rows
    }


def add_column_if_missing(
    connection,
    table_name,
    column_name,
    column_definition
):
    columns = get_columns(
        connection,
        table_name
    )

    if column_name in columns:
        print(
            f"[EXISTS] "
            f"{table_name}.{column_name}"
        )
        return False

    sql = (
        f"ALTER TABLE {table_name} "
        f"ADD COLUMN {column_name} "
        f"{column_definition}"
    )

    connection.execute(sql)

    print(
        f"[ADDED] "
        f"{table_name}.{column_name} "
        f"-> {column_definition}"
    )

    return True


def verify_table(connection, table_name):
    print()
    print("=" * 70)
    print(f"TABLE: {table_name}")
    print("=" * 70)

    rows = connection.execute(
        f"PRAGMA table_info({table_name})"
    ).fetchall()

    for row in rows:
        print(
            f"{row['name']:30} "
            f"{row['type']:15} "
            f"NOT NULL={row['notnull']} "
            f"DEFAULT={row['dflt_value']}"
        )


def main():
    print("=" * 70)
    print("PAPRIKA - LEAF TRACKING DATABASE MIGRATION")
    print("=" * 70)

    print()
    print(f"Project root:")
    print(PROJECT_ROOT)

    print()
    print(f"Database:")
    print(DATABASE_PATH)

    if not DATABASE_PATH.exists():
        print()
        print("[ERROR] Database does not exist.")
        print(DATABASE_PATH)
        return 1

    connection = get_connection()

    try:
        # ----------------------------------------------------
        # Verify required tables
        # ----------------------------------------------------

        tables = {
            row["name"]
            for row in connection.execute(
                """
                SELECT name
                FROM sqlite_master
                WHERE type = 'table'
                """
            ).fetchall()
        }

        required_tables = {
            "segmentation_runs",
            "leaf_objects",
        }

        missing_tables = (
            required_tables - tables
        )

        if missing_tables:
            print()
            print(
                "[ERROR] Missing required tables:"
            )

            for table in sorted(missing_tables):
                print(f"  - {table}")

            return 1

        # ----------------------------------------------------
        # segmentation_runs
        # ----------------------------------------------------

        print()
        print(
            "Updating segmentation_runs..."
        )

        add_column_if_missing(
            connection,
            "segmentation_runs",
            "detected_leaf_count",
            "INTEGER DEFAULT 0"
        )

        add_column_if_missing(
            connection,
            "segmentation_runs",
            "selected_leaf_count",
            "INTEGER DEFAULT 0"
        )

        add_column_if_missing(
            connection,
            "segmentation_runs",
            "selection_created_at",
            "TEXT"
        )

        # ----------------------------------------------------
        # leaf_objects
        # ----------------------------------------------------

        print()
        print(
            "Updating leaf_objects..."
        )

        add_column_if_missing(
            connection,
            "leaf_objects",
            "check_status",
            "TEXT NOT NULL DEFAULT 'pending'"
        )

        add_column_if_missing(
            connection,
            "leaf_objects",
            "checked_at",
            "TEXT"
        )

        add_column_if_missing(
            connection,
            "leaf_objects",
            "check_count",
            "INTEGER NOT NULL DEFAULT 0"
        )

        # ----------------------------------------------------
        # Normalize existing rows
        # ----------------------------------------------------
        #
        # Existing leaves should start as pending.
        # Existing runs should have zero selection until
        # the new selection mechanism is used.
        # ----------------------------------------------------

        print()
        print(
            "Normalizing existing records..."
        )

        connection.execute(
            """
            UPDATE leaf_objects
            SET check_status = 'pending'
            WHERE check_status IS NULL
               OR TRIM(check_status) = ''
            """
        )

        connection.execute(
            """
            UPDATE leaf_objects
            SET check_count = 0
            WHERE check_count IS NULL
            """
        )

        connection.execute(
            """
            UPDATE segmentation_runs
            SET detected_leaf_count = 0
            WHERE detected_leaf_count IS NULL
            """
        )

        connection.execute(
            """
            UPDATE segmentation_runs
            SET selected_leaf_count = 0
            WHERE selected_leaf_count IS NULL
            """
        )

        # ----------------------------------------------------
        # Indexes
        # ----------------------------------------------------

        print()
        print(
            "Creating indexes..."
        )

        connection.execute(
            """
            CREATE INDEX IF NOT EXISTS
            idx_leaf_objects_run_status
            ON leaf_objects(
                run_id,
                check_status
            )
            """
        )

        connection.execute(
            """
            CREATE INDEX IF NOT EXISTS
            idx_leaf_objects_checked_at
            ON leaf_objects(
                checked_at
            )
            """
        )

        connection.commit()

        # ----------------------------------------------------
        # Verification
        # ----------------------------------------------------

        verify_table(
            connection,
            "segmentation_runs"
        )

        verify_table(
            connection,
            "leaf_objects"
        )

        # ----------------------------------------------------
        # Summary
        # ----------------------------------------------------

        print()
        print("=" * 70)
        print("MIGRATION COMPLETED SUCCESSFULLY")
        print("=" * 70)

        print()
        print(
            "Added to segmentation_runs:"
        )
        print(
            "  - detected_leaf_count"
        )
        print(
            "  - selected_leaf_count"
        )
        print(
            "  - selection_created_at"
        )

        print()
        print(
            "Added to leaf_objects:"
        )
        print(
            "  - check_status"
        )
        print(
            "  - checked_at"
        )
        print(
            "  - check_count"
        )

        print()
        print(
            "Existing data was NOT deleted."
        )

        return 0

    except Exception as exc:
        connection.rollback()

        print()
        print("=" * 70)
        print("MIGRATION FAILED")
        print("=" * 70)
        print()
        print(str(exc))

        return 1

    finally:
        connection.close()


if __name__ == "__main__":
    raise SystemExit(main())